import os
import streamlit as st
import google.generativeai as genai
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

# Configure GenAI and CrewAI LLM
clean_key = api_key.strip()
genai.configure(api_key=clean_key)

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

    with st.spinner("Finding available model & extracting data via Gemini..."):
        try:
            # Dynamically find the supported model for this API key
            available_models = [
                m.name for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            ]
            
            # Select flash or pro if available, else first supported
            chosen_model_name = "models/gemini-1.5-flash"
            for candidate in ["models/gemini-1.5-flash", "models/gemini-1.5-flash-8b", "models/gemini-1.5-pro", "models/gemini-pro"]:
                if candidate in available_models:
                    chosen_model_name = candidate
                    break
            else:
                chosen_model_name = available_models[0] if available_models else "gemini-1.5-flash"

            model = genai.GenerativeModel(chosen_model_name)
            
            prompt = (
                "Analyze these attached property claims photos, handwritten/printed scope notes, and EagleView reports. "
                "Extract: 1) Roof pitch, squares, facets, ridges, hips, valleys, eaves, rakes. "
                "2) Directional slope damage hit counts / test squares. "
                "3) Exterior elevation collateral damage (gutters, siding, screens, soft metals). "
                "4) Any building code mandates noted."
            )

            content_list = [prompt]

            if uploaded_pdf:
                pdf_bytes = uploaded_pdf.read()
                content_list.append({"mime_type": "application/pdf", "data": pdf_bytes})

            if uploaded_images:
                for img_file in uploaded_images:
                    img = Image.open(img_file).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    content_list.append({"mime_type": "image/jpeg", "data": buf.getvalue()})

            extraction_response = model.generate_content(content_list)
            extracted_data = extraction_response.text
            st.success(f"✅ Data extracted using {chosen_model_name}!")

        except Exception as e:
            st.error(f"Error reading documents/photos: {e}")
            st.stop()

    # Run CrewAI Pipeline
    with st.spinner("Crew is calculating thresholds, checking IRC code, and scoping Xactimate line items..."):
        try:
            crew_llm = LLM(model=f"gemini/{chosen_model_name.replace('models/', '')}", api_key=clean_key)

            gilligan = Agent(
                role="Field Claims Inspector (Gilligan)",
                goal="Evaluate damage counts against the 6-hit threshold per slope and 50% roof replacement rule",
                backstory="Experienced field adjuster who verifies hail/wind damage thresholds and drafts inspection narratives.",
                llm=crew_llm,
                verbose=False
            )

            ginger = Agent(
                role="Residential Building Code Specialist (Ginger)",
                goal="Identify mandatory 2021/2024 IRC Chapter 9 provisions (drip edge, underlayment, crickets >30in, valleys)",
                backstory="Forensic building code specialist ensuring all mandatory code upgrades are included.",
                llm=crew_llm,
                verbose=False
            )

            estimator = Agent(
                role="Certified Xactimate Estimator",
                goal="Generate a line-by-line itemized Xactimate schedule with CAT/SEL codes, quantities, and F9 notes",
                backstory="Expert property loss estimator providing comprehensive line items without skipping ancillary scope.",
                llm=crew_llm,
                verbose=False
            )

            t1 = Task(
                description=f"Analyze extracted inspection facts for {cause_of_loss} claim: {extracted_data}. Apply the rule: 6+ hits per test square replaces the slope; 50%+ slopes replaced triggers full roof replacement.",
                expected_output="Adjuster Narrative Report detailing slope breakdown and replacement determination.",
                agent=gilligan
            )

            t2 = Task(
                description="Review extracted field facts and identify required IRC/IBC provisions (drip edge perimeter, ice barrier SF, valley metal, step flashing, chimney crickets).",
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
