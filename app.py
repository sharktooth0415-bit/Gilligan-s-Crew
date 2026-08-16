import os
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
from crewai import Agent, Task, Crew, Process, LLM

st.set_page_config(page_title="Claims Multi-Agent App", page_icon="📸", layout="centered")
st.title("📸 Field Claims Scoping Crew")

# Retrieve API Key
api_key = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to get started.")
    st.stop()

# Initialize Gemini Client and CrewAI LLM with gemini-1.5-flash
client = genai.Client(api_key=api_key.strip())
llm = LLM(model="gemini/gemini-1.5-flash", api_key=api_key.strip())

st.subheader("1. Upload Documents & Field Photos")
uploaded_pdf = st.file_uploader("Upload EagleView Report (PDF)", type=["pdf"])
uploaded_images = st.file_uploader(
    "Take Photos or Upload Scope Sheets / Damage Notes",
    type=["jpg", "jpeg", "png", "webp", "heic"],
    accept_multiple_files=True
)

cause_of_loss = st.selectbox("Cause of Loss", ["Hail", "Wind", "Water"])

if st.button("Analyze Photos & Run Crew 🚀", type="primary"):
    if not uploaded_images and not uploaded_pdf:
        st.warning("Please upload at least one photo, scope sheet, or EagleView PDF.")
        st.stop()

    with st.spinner("Processing documents & images via Gemini Vision..."):
        try:
            vision_parts = [
                types.Part.from_text(
                    text=(
                        "Analyze these attached property claims photos, handwritten/printed scope notes, and EagleView reports. "
                        "Extract: 1) Roof pitch, squares, facets, ridges, hips, valleys, eaves, rakes. "
                        "2) Directional slope damage hit counts / test squares. "
                        "3) Exterior elevation collateral damage (gutters, siding, screens, soft metals). "
                        "4) Any building code mandates noted."
                    )
                )
            ]

            # Handle PDF uploads
            if uploaded_pdf:
                pdf_bytes = uploaded_pdf.read()
                vision_parts.append(
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
                )

            # Handle Image uploads
            if uploaded_images:
                for img_file in uploaded_images:
                    img = Image.open(img_file).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    vision_parts.append(
                        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
                    )

            # Extract data using gemini-1.5-flash
            extraction_response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=vision_parts
            )
            extracted_data = extraction_response.text
            st.success("✅ Data extracted from documents successfully!")

        except Exception as e:
            st.error(f"Error reading documents/photos: {e}")
            st.stop()

    # Run CrewAI Pipeline
    with st.spinner("Crew is calculating thresholds, checking IRC code, and scoping Xactimate line items..."):
        try:
            gilligan = Agent(
                role="Field Claims Inspector (Gilligan)",
                goal="Evaluate damage counts against the 6-hit threshold per slope and 50% roof replacement rule",
                backstory="Experienced field adjuster who verifies hail/wind damage thresholds and drafts inspection narratives.",
                llm=llm,
                verbose=False
            )

            ginger = Agent(
                role="Residential Building Code Specialist (Ginger)",
                goal="Identify mandatory 2021/2024 IRC Chapter 9 provisions (drip edge, underlayment, crickets >30in, valleys)",
                backstory="Forensic building code specialist ensuring all mandatory code upgrades are included.",
                llm=llm,
                verbose=False
            )

            estimator = Agent(
                role="Certified Xactimate Estimator",
                goal="Generate a line-by-line itemized Xactimate schedule with CAT/SEL codes, quantities, and F9 notes",
                backstory="Expert property loss estimator providing comprehensive line items without skipping ancillary scope.",
                llm=llm,
                verbose=False
            )

            t1 = Task(
                description=f"Analyze extracted inspection facts for {cause_of_loss} claim: {extracted_data}. Apply the rule: 6+ hits per test square replaces the slope; 50%+ slopes replaced triggers full roof replacement.",
                expected_output="Adjuster Narrative Report detailing slope breakdown and replacement determination.",
                agent=gilligan
            )

            t2 = Task(
                description=f"Review extracted field facts and identify required IRC/IBC provisions (drip edge perimeter, ice barrier SF, valley metal, step flashing, chimney crickets).",
                expected_output="Code compliance breakdown citing IRC sections.",
                agent=ginger
            )

            t3 = Task(
                description="Using Gilligan's narrative and Ginger's code report, produce a complete itemized Xactimate schedule with Category, Selector, Action, Description, Qty, Unit, and F9 justification notes.",
                expected_output="Markdown table with full Xactimate line items.",
                agent=estimator
            )

            claim_crew = Crew(
                agents=[gilligan, ginger, estimator],
                tasks=[t1, t2, t3],
                process=Process.sequential
            )

            crew_output = claim_crew.kickoff()

            st.subheader("📋 Final Multi-Agent Scope & Narrative")
            st.markdown(crew_output.raw)

        except Exception as e:
            st.error(f"Error during CrewAI execution: {e}")
