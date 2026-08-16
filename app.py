import os
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import PDFSearchTool # <-- Added this back in

st.set_page_config(page_title="Claims Multi-Agent App", page_icon="⚓", layout="centered")
st.title("⚓ Skipper's Claims Crew")

api_key = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to get started.")
    st.stop()

clean_key = api_key.strip()
os.environ["GEMINI_API_KEY"] = clean_key

client = genai.Client(api_key=clean_key)
llm = LLM(model="gemini/gemini-3.5-flash", api_key=clean_key)

# --- LOAD THE XACTIMATE PDFS ---
# Put all your chunked PDFs into a folder named "xactimate_pdfs" right next to app.py
xactimate_tool = PDFSearchTool(directory="xactimate_pdfs")

# --- INITIALIZE THE ENTIRE CREW ---
gilligan = Agent(
    role="Field Claims Inspector (Gilligan)",
    goal="Evaluate damage counts against the 6-hit threshold per slope and 50% roof replacement rule.",
    backstory="Experienced field adjuster who verifies hail/wind damage thresholds and drafts inspection narratives.",
    llm=llm,
    allow_delegation=False,
    verbose=False
)

ginger = Agent(
    role="Residential Building Code Specialist (Ginger)",
    goal="Identify mandatory 2021/2024 IRC Chapter 9 provisions.",
    backstory="Forensic building code specialist ensuring all mandatory code upgrades are included. You know exactly what code applies to roofing components.",
    llm=llm,
    allow_delegation=False,
    verbose=False
)

# --- XACTIMATE MASTER CATEGORY INDEX ---
XACTIMATE_MASTER_INDEX = """
ACC: ACCESSORIES - MOBILE HOME
ACT: ACOUSTICAL TREATMENTS
APP: APPLIANCES
AWN: AWNINGS & PATIO COVERS
CAB: CABINETRY
CAP: CONT: CLEAN APPLIANCES
CDC: CONT: GARMENT & SOFT GOODS CLN
CEL: CONT: CLEAN ELECTRIC ITEMS
CGN: CONT: CLEAN - GENERAL ITEMS
CHF: CONT: CLEAN - HARD FURNITURE
CLM: CONT: CLEAN - LAMPS OR VASES
CLN: CLEANING
CNC: CONCRETE & ASPHALT
CON: CONTENT MANIPULATION
CPS: CONT: PACKING,HANDLNG,STORAGE
CUP: CONT: CLEAN, UPHOLSTERY & SOFT
CWH: CONT: CEILING/WALL HANGINGS
DMO: GENERAL DEMOLITION
DOR: DOORS
DRY: DRYWALL
ELE: ELECTRICAL
ELS: ELECTRICAL - SPECIAL SYSTEMS
EQA: MISC. EQUIPMENT - AGRICULTURAL
EQC: MISC. EQUIPMENT - COMMERCIAL
EQU: HEAVY EQUIPMENT
EXC: EXCAVATION
FCC: FLOOR COVERING - CARPET
FCR: FLOOR COVERING - RESILIENT
FCS: FLOOR COVERING - STONE
FCT: FLOOR COVERING - CERAMIC TILE
FCV: FLOOR COVERING - VINYL
FCW: FLOOR COVERING - WOOD
FEE: PERMITS AND FEES
FEN: FENCING
FNC: FINISH CARPENTRY / TRIMWORK
FNH: FINISH HARDWARE
FPL: FIREPLACES
FPS: FIRE PROTECTION SYSTEMS
FRM: FRAMING & ROUGH CARPENTRY
GLS: GLASS, GLAZING, & STOREFRONTS
HMR: HAZARDOUS MATERIAL REMEDIATION
HVC: HEAT, VENT & AIR CONDITIONING
INM: INSULATION - MECHANICAL
INS: INSULATION
LAB: LABOR ONLY
LIT: LIGHT FIXTURES
LND: LANDSCAPING
MAS: MASONRY
MBL: MARBLE - CULTURED OR NATURAL
MPR: MOISTURE PROTECTION
MSD: MIRRORS & SHOWER DOORS
MSK: MOBILE HOMES, SKIRTING & SETUP
MTL: METAL STRUCTURES & COMPONENTS
ORI: ORNAMENTAL IRON
PLA: INTERIOR LATH & PLASTER
PLM: PLUMBING
PNL: PANELING & WOOD WALL FINISHES
PNT: PAINTING
POL: SWIMMING POOLS & SPAS
RFG: ROOFING
SCF: SCAFFOLDING
SDG: SIDING
SFG: SOFFIT, FASCIA, & GUTTER
SPE: SPECIALTY ITEMS
STJ: STEEL JOIST COMPONENTS
STL: STEEL COMPONENTS
STR: STAIRS
STU: STUCCO & EXTERIOR PLASTER
TBA: TOILET & BATH ACCESSORIES
TCR: TRAUMA/CRIME SCENE REMEDIATION
TIL: TILE
TMB: TIMBER FRAMING
TMP: TEMPORARY REPAIRS
USR: USER DEFINED ITEMS
WDA: WINDOWS - ALUMINUM
WDP: WINDOWS - SLIDING PATIO DOORS
WDR: WINDOW REGLAZING & REPAIR
WDS: WINDOWS - SKYLIGHTS
WDT: WINDOW TREATMENT
WDV: WINDOWS - VINYL
WDW: WINDOWS - WOOD
WPR: WALLPAPER
WTR: WATER EXTRACTION & REMEDIATION
XST: EXTERIOR STRUCTURES
"""

professor = Agent(
    role="Certified Xactimate Estimator (The Professor)",
    goal="Generate Xactimate line items using exact CAT codes and descriptions from the provided PDF reference files.",
    backstory=f"Expert property loss estimator. You MUST use your PDF search tool to find the correct Xactimate CAT and SEL codes. NEVER guess or make up a code. If it is not in your search results, leave the SEL blank or state you can't find it. ALWAYS reference this master index to ensure you are using the correct 3-letter CAT code for the trade:\n\n{XACTIMATE_MASTER_INDEX}",
    llm=llm,
    tools=[xactimate_tool],
    allow_delegation=False,
    verbose=False
)

skipper = Agent(
    role="Crew Manager (The Skipper)",
    goal="Analyze the user's input. Decide if it's a simple question or a full estimate request, and delegate to the crew accordingly.",
    backstory="You are the captain. You know your crew's strengths: Gilligan handles field facts, Ginger handles building codes, and The Professor handles Xactimate codes. If the user provides a simple question, delegate to the right expert. If the user provides field notes/photos for a full claim, you MUST delegate to all three experts to compile a full comprehensive report.",
    llm=llm,
    allow_delegation=True,
    verbose=False
)

# --- UNIFIED UI ---
st.subheader("What do you need the crew to do?")
user_input = st.text_area("Ask a question, or type 'Write a full estimate' if uploading files:")

uploaded_files = st.file_uploader(
    "Upload EagleView (PDF or Photo) & Scope Sheets (Optional)",
    type=["pdf", "jpg", "jpeg", "png", "webp", "heic"],
    accept_multiple_files=True
)

cause_of_loss = st.selectbox("Cause of Loss (if applicable)", ["Hail", "Wind", "Water", "None"])

if st.button("Engage Crew 🚀", type="primary"):
    if not user_input and not uploaded_files:
        st.warning("Please type a request or upload some documents/photos.")
        st.stop()

    extracted_data = "No files uploaded."

    if uploaded_files:
        with st.spinner("Processing documents & images via Gemini Vision..."):
            try:
                prompt_text = (
                    "Analyze these attached property claims photos, handwritten/printed scope notes, and EagleView reports. "
                    "Extract: 1) Roof pitch, squares, facets, ridges, hips, valleys, eaves, rakes. "
                    "2) Directional slope damage hit counts / test squares. "
                    "3) Exterior elevation collateral damage. "
                    "4) Any building code mandates noted."
                )
                
                vision_parts = [types.Part.from_text(text=prompt_text)]

                for file in uploaded_files:
                    if file.name.lower().endswith(".pdf"):
                        pdf_bytes = file.read()
                        vision_parts.append(types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))
                    else:
                        img = Image.open(file).convert("RGB")
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG")
                        vision_parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))

                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=vision_parts
                )
                extracted_data = response.text
                st.success("✅ File data extracted successfully!")
            except Exception as e:
                st.error(f"Error reading documents/photos: {e}")
                st.stop()

    with st.spinner("Skipper is reviewing the request and coordinating the crew..."):
        try:
            master_task = Task(
                description=f"User's Request: '{user_input}'\n\nCause of Loss: {cause_of_loss}\n\nExtracted File Data: {extracted_data}\n\nInstructions: Analyze the request and the data. If it is a simple question, delegate it to the appropriate expert and return their answer. If it requires a full estimate based on the extracted data, delegate to Gilligan, Ginger, and the Professor to compile a full comprehensive report.",
                expected_output="A direct answer to the user's question, OR a fully compiled inspection, code, and estimating report.",
                agent=skipper
            )

            claim_crew = Crew(
                agents=[skipper, gilligan, ginger, professor],
                tasks=[master_task],
                process=Process.sequential
            )

            crew_output = claim_crew.kickoff()

            st.subheader("📋 Final Output")
            st.markdown(crew_output.raw)

        except Exception as e:
            st.error(f"Error during CrewAI execution: {e}")
