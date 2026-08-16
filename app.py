import os
import streamlit as st
from PIL import Image
from google import genai
from crewai import Agent, Task, Crew, Process, LLM

st.set_page_config(page_title="Claims Multi-Agent App", page_icon="📸", layout="centered")
st.title("📸 Field Claims Scoping Crew")

# Configure API Key
api_key = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to get started.")
    st.stop()

# Initialize Gemini Client for Direct Vision Extraction
client = genai.Client(api_key=api_key)
llm = LLM(model="gemini/gemini-2.0-flash", api_key=api_key)

# Mobile File / Photo Uploaders
st.subheader("1. Upload Documents & Field Photos")
eagleview_pdf = st.file_uploader("Upload EagleView Report (PDF)", type=["pdf"])
uploaded_images = st.file_uploader(
    "Take Photos or Upload Scope Sheets / Damage Notes",
    type=["jpg", "jpeg", "png", "webp", "heic"],
    accept_multiple_files=True
)

cause_of_loss = st.selectbox("Cause of Loss", ["Hail", "Wind", "Water"])

if st.button("Analyze Photos & Run Crew 🚀", type="primary"):
    if not uploaded_images and not eagleview_pdf:
        st.warning("Please upload at least one photo, scope sheet, or EagleView PDF.")
        st.stop()

    with st.spinner("Processing photos & extracting document data via Gemini Vision..."):
        # Process uploaded images/PDFs into Gemini Vision parts
        vision_contents = [
            "Analyze these attached property claims photos, handwritten/printed scope notes, and EagleView reports. "
            "Extract: 1) Roof pitch, squares, facets, ridges, hips, valleys, eaves, rakes. "
            "2) Directional slope damage hit counts / test squares. "
            "3) Exterior elevation collateral damage (gutters, siding, screens, soft metals). "
            "4) Any code mandates noted."
        ]

        if uploaded_images:
            for img_file in uploaded_images:
                img = Image.open(img_file)
                vision_contents.append(img)

        # Gemini 2.0 Flash Vision Extraction
        extraction_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=vision_contents
        )
        extracted_data = extraction_response.text

    st.success("✅ Document data extracted. Initializing CrewAI...")

    # CrewAI Multi-Agent Pipeline
    with st.spinner("Crew is calculating thresholds, checking IRC code, and scoping Xactimate line items..."):
        
        # Agent 1: Gilligan (Field Inspector)
        gilligan = Agent(
            role="Field Claims Inspector (Gilligan)",
            goal="Evaluate damage counts against the 6-hit threshold per slope and 50% roof replacement rule",
            backstory="Experienced field adjuster who verifies hail/wind damage thresholds and drafts inspection narratives.",
            llm=llm,
            verbose=False
        )

        # Agent 2: Ginger (Building Code Specialist)
        ginger = Agent(
            role="Residential Building Code Specialist (Ginger)",
            goal="Identify mandatory 2021/2024 IRC Chapter 9 provisions (drip edge, underlayment, crickets >30in, valleys)",
            backstory="Forensic building code specialist ensuring all mandatory code upgrades are included.",
            llm=llm,
            verbose=False
        )

        # Agent 3: Estimator (Xactimate Estimator)
        estimator = Agent(
            role="Certified Xactimate Estimator",
            goal="Generate a line-by-line itemized Xactimate schedule with CAT/SEL codes, quantities, and F9 notes",
            backstory="Expert property loss estimator providing comprehensive line items without skipping ancillary scope.",
            llm=llm,
            verbose=False
        )

        # Tasks
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
