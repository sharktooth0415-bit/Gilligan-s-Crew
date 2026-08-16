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

clean_key = api_key.strip()
os.environ["GEMINI_API_KEY"] = clean_key

# Initialize modern GenAI client with current 3.5 flash model
client = genai.Client(api_key=clean_key)
llm = LLM(model="gemini/gemini-3.5-flash", api_key=clean_key)

# --- THE PROFESSOR AGENT (Updated for Clarification) ---
professor = Agent(
    role="Certified Xactimate Estimator (The Professor)",
    goal="Generate Xactimate line items OR explicitly ask clarifying questions if details are missing or ambiguous.",
    backstory="Expert property loss estimator. You never guess, assume, or hallucinate Selector (SEL) codes or dimensions. If a scope sheet is missing measurements, lacks context, or is ambiguous, you stop the estimation process and output a numbered list of the exact questions you need the user to answer before you can proceed.",
    llm=llm,
    verbose=False
)

# --- NEW BYPASS SECTION ---
st.subheader("1. Ask The Professor Directly (Bypass Full Scope)")
direct_question = st.text_area("Have a quick Xactimate line item question? Ask here:")
if st.button("Ask The Professor 🧠"):
    if not direct_question:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("The Professor is digging through his memory..."):
            try:
                t_direct = Task(
                    description=f"Answer the following Xactimate estimating question: {direct_question}. Provide the exact CAT code and line item description. Only provide the SEL code if you are absolutely sure, otherwise leave it blank.",
                    expected_output="Direct answer containing the requested Xactimate line items or advice.",
                    agent=professor
                )
                direct_crew = Crew(agents=[professor], tasks=[t_direct])
                result = direct_crew.kickoff()
                st.info("📋 **The Professor's Answer:**")
                st.markdown(result.raw)
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()

# --- ORIGINAL FULL CREW SECTION ---
st.subheader("2. Full Multi-Agent Scoping")
uploaded_files = st.file_uploader(
    "Upload EagleView (PDF or Photo) & Scope Sheets / Damage Notes",
    type=["pdf", "jpg", "jpeg", "png", "webp", "heic"],
    accept_multiple_files=True
)

cause_of_loss = st.selectbox("Cause of Loss", ["Hail", "Wind", "Water"])

if st.button("Analyze Photos & Run Full Crew 🚀", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one photo, scope sheet, or EagleView document.")
        st.stop()

    with st.spinner("Processing documents & images via Gemini Vision..."):
        try:
            prompt_text = (
                "Analyze these attached property claims photos, handwritten/printed scope notes, and EagleView reports. "
                "Extract: 1) Roof pitch, squares, facets, ridges, hips, valleys, eaves, rakes. "
                "2) Directional slope damage hit counts / test squares. "
                "3) Exterior elevation collateral damage (gutters, siding, screens, soft metals). "
                "4) Any building code mandates noted."
            )
            
            vision_parts = [types.Part.from_text(text=prompt_text)]

            for file in uploaded_files:
                if file.name.lower().endswith(".pdf"):
                    pdf_bytes = file.read()
                    vision_parts.append(
                        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
                    )
                else:
                    img = Image.open(file).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    vision_parts.append(
                        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
                    )

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=vision_parts
            )
            extracted_data = response.text
            st.success("✅ Data extracted successfully!")

        except Exception as e:
            st.error(f"Error reading documents/photos: {e}")
            st.stop()

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
                description="Using Gilligan's narrative and Ginger's code report, produce a complete itemized Xactimate schedule. Provide the exact CAT code and description. Leave the SEL code blank unless you are 100% certain.",
                expected_output="Markdown table with full Xactimate line items.",
                agent=professor
            )

            claim_crew = Crew(
                agents=[gilligan, ginger, professor],
                tasks=[t1, t2, t3],
                process=Process.sequential
            )

            crew_output = claim_crew.kickoff()

            st.subheader("🧢 Gilligan's Inspection Narrative")
            st.markdown(t1.output.raw)

            st.subheader("👠 Ginger's Code Compliance Report")
            st.markdown(t2.output.raw)

            st.subheader("📋 The Professor's Xactimate Schedule")
            st.markdown(t3.output.raw)

        except Exception as e:
            st.error(f"Error during CrewAI execution: {e}")
