import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import PDFSearchTool


# Optional HEIC support. The application will still run without it,
# but HEIC uploads require pillow-heif in requirements.txt.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Claims Multi-Agent App",
    page_icon="⚓",
    layout="centered",
)

st.title("⚓ Skipper's Claims Crew")


# ============================================================
# GEMINI API KEY
# ============================================================

environment_api_key = os.environ.get("GEMINI_API_KEY", "")

api_key = environment_api_key or st.sidebar.text_input(
    "Gemini API Key",
    type="password",
)

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to get started.")
    st.stop()

clean_key = api_key.strip()

if not clean_key:
    st.warning("The Gemini API key cannot be blank.")
    st.stop()

# Different libraries look for different Google key names.
os.environ["GEMINI_API_KEY"] = clean_key
os.environ["GOOGLE_API_KEY"] = clean_key

client = genai.Client(api_key=clean_key)

llm = LLM(
    model="gemini/gemini-3.5-flash",
    api_key=clean_key,
)


# ============================================================
# XACTIMATE PDF SEARCH TOOLS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
XACTIMATE_DIR = APP_DIR / "xactimate_pdfs"


def make_pdf_config(api_key_value: str) -> dict:
    """
    Configure PDFSearchTool to use Gemini embeddings instead of
    its default OpenAI embedding provider.
    """
    return {
        "embedding_model": {
            "provider": "google-generativeai",
            "config": {
                "model_name": "gemini-embedding-001",
                "api_key": api_key_value,
                "task_type": "RETRIEVAL_DOCUMENT",
                "title": "Xactimate Reference",
            },
        },
        "vectordb": {
            "provider": "chromadb",
            "config": {},
        },
    }


@st.cache_resource(
    show_spinner="Indexing Xactimate reference PDFs. This may take a few minutes..."
)
def load_xactimate_tools(api_key_value: str):
    """
    Find every PDF in xactimate_pdfs and create a separate search
    tool for each document.
    """
    if not XACTIMATE_DIR.exists():
        raise FileNotFoundError(
            f"The Xactimate PDF directory does not exist: {XACTIMATE_DIR}"
        )

    if not XACTIMATE_DIR.is_dir():
        raise NotADirectoryError(
            f"The Xactimate PDF path is not a directory: {XACTIMATE_DIR}"
        )

    pdf_files = sorted(XACTIMATE_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in: {XACTIMATE_DIR}"
        )

    tools = []

    for pdf_file in pdf_files:
        tool = PDFSearchTool(
            pdf=str(pdf_file),
            config=make_pdf_config(api_key_value),
        )
        tools.append(tool)

    return tools


try:
    xactimate_tools = load_xactimate_tools(clean_key)
except Exception as error:
    st.error(
        "Could not load the Xactimate reference PDFs.\n\n"
        f"Technical details: {error}"
    )
    st.info(
        "Confirm that the xactimate_pdfs folder and its PDF files "
        "were committed to your repository."
    )
    st.stop()


# ============================================================
# XACTIMATE MASTER CATEGORY INDEX
# ============================================================

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
CPS: CONT: PACKING, HANDLING, STORAGE
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


# ============================================================
# AGENTS
# ============================================================

gilligan = Agent(
    role="Field Claims Inspector (Gilligan)",
    goal=(
        "Evaluate documented property damage, including damage counts, "
        "test squares, affected slopes, and replacement considerations."
    ),
    backstory=(
        "You are an experienced field adjuster who evaluates hail, wind, "
        "water, and exterior damage. Clearly distinguish documented facts "
        "from assumptions. Do not invent measurements or damage."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=False,
)


ginger = Agent(
    role="Residential Building Code Specialist (Ginger)",
    goal=(
        "Identify potentially applicable residential building-code provisions "
        "and explain what additional jurisdictional verification is needed."
    ),
    backstory=(
        "You are a building-code specialist familiar with the 2021 and 2024 "
        "IRC. Never claim that a provision is mandatory without identifying "
        "the applicable edition, jurisdiction, and factual trigger. Clearly "
        "label anything that requires confirmation by the local authority "
        "having jurisdiction."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=False,
)


professor = Agent(
    role="Certified Xactimate Estimator (The Professor)",
    goal=(
        "Generate Xactimate line-item recommendations using exact CAT codes "
        "and descriptions found in the provided PDF reference files."
    ),
    backstory=f"""
You are an expert property-loss estimator.

You MUST use the available PDF search tools to locate the correct Xactimate
CAT and SEL codes.

Rules:

1. Never invent a CAT code, SEL code, description, unit, or price.
2. If a SEL code cannot be verified in the PDF references, leave it blank
   or state that it could not be found.
3. Clearly separate verified Xactimate information from recommendations.
4. Use the following master category index to select the appropriate trade.
5. When possible, identify which reference PDF supports the recommendation.

Xactimate master category index:

{XACTIMATE_MASTER_INDEX}
""",
    llm=llm,
    tools=xactimate_tools,
    allow_delegation=False,
    verbose=False,
)


skipper = Agent(
    role="Crew Manager (The Skipper)",
    goal=(
        "Analyze the user's request, coordinate the appropriate specialists, "
        "and return a clear, accurate final response."
    ),
    backstory=(
        "You are the crew manager. Gilligan handles field observations, "
        "Ginger handles building-code considerations, and The Professor "
        "handles Xactimate codes and line-item research. For simple questions, "
        "delegate to the appropriate specialist. For full claim or estimate "
        "requests, consult all relevant specialists and combine their findings. "
        "Never present assumptions as established facts."
    ),
    llm=llm,
    allow_delegation=True,
    verbose=False,
)


# ============================================================
# USER INTERFACE
# ============================================================

st.subheader("What do you need the crew to do?")

user_input = st.text_area(
    "Ask a question, or type “Write a full estimate” if uploading files:"
)

uploaded_files = st.file_uploader(
    "Upload EagleView reports, scope sheets, or property photos",
    type=["pdf", "jpg", "jpeg", "png", "webp", "heic"],
    accept_multiple_files=True,
)

cause_of_loss = st.selectbox(
    "Cause of Loss",
    ["Hail", "Wind", "Water", "Fire", "Other", "None"],
)

if uploaded_files:
    st.caption(f"{len(uploaded_files)} file(s) selected.")


# ============================================================
# FILE EXTRACTION
# ============================================================

def build_uploaded_file_parts(files):
    """
    Convert uploaded PDFs and images into Gemini input parts.
    """
    prompt_text = (
        "Analyze the attached property-claim materials. They may include "
        "property photos, handwritten or printed scope notes, and EagleView "
        "reports.\n\n"
        "Extract only information supported by the files:\n"
        "1. Roof pitch, squares, facets, ridges, hips, valleys, eaves, and rakes.\n"
        "2. Directional slope damage counts and test-square observations.\n"
        "3. Exterior elevation and collateral damage.\n"
        "4. Interior damage and affected rooms or materials.\n"
        "5. Any building-code provisions explicitly mentioned.\n"
        "6. Any quantities, measurements, or scope notes relevant to estimating.\n\n"
        "If a value is not visible or cannot be verified, state that it could "
        "not be determined. Do not guess."
    )

    parts = [types.Part.from_text(text=prompt_text)]

    for uploaded_file in files:
        filename = uploaded_file.name.lower()

        if filename.endswith(".pdf"):
            pdf_bytes = uploaded_file.getvalue()

            parts.append(
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                )
            )
            continue

        if filename.endswith(".heic") and not HEIC_SUPPORTED:
            raise RuntimeError(
                f"HEIC support is not installed, so {uploaded_file.name} "
                "cannot be processed. Add pillow-heif to requirements.txt "
                "or upload the image as JPG or PNG."
            )

        uploaded_file.seek(0)

        with Image.open(uploaded_file) as source_image:
            image = source_image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)

        parts.append(
            types.Part.from_bytes(
                data=buffer.getvalue(),
                mime_type="image/jpeg",
            )
        )

    return parts


def extract_uploaded_file_data(files):
    """
    Send the uploaded files to Gemini for multimodal extraction.
    """
    parts = build_uploaded_file_parts(files)

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Content(
                role="user",
                parts=parts,
            )
        ],
    )

    if not response or not response.text:
        raise RuntimeError(
            "Gemini did not return extracted text for the uploaded files."
        )

    return response.text


# ============================================================
# RUN THE CREW
# ============================================================

if st.button("Engage Crew 🚀", type="primary"):
    if not user_input.strip() and not uploaded_files:
        st.warning(
            "Please enter a request or upload at least one document or photo."
        )
        st.stop()

    extracted_data = "No files were uploaded."

    if uploaded_files:
        with st.spinner("Processing documents and images with Gemini..."):
            try:
                extracted_data = extract_uploaded_file_data(uploaded_files)
                st.success("File data extracted successfully.")
            except Exception as error:
                st.error(
                    "The uploaded documents or images could not be processed."
                )
                st.exception(error)
                st.stop()

    request_text = user_input.strip() or (
        "Review the uploaded claim materials and summarize the documented scope."
    )

    with st.spinner(
        "Skipper is reviewing the request and coordinating the crew..."
    ):
        try:
            master_task = Task(
                description=f"""
User request:

{request_text}

Cause of loss:

{cause_of_loss}

Extracted file data:

{extracted_data}

Instructions:

1. Determine whether this is a simple question, a document review, or a full
   claim-estimate request.
2. Delegate field-damage analysis to Gilligan when appropriate.
3. Delegate building-code analysis to Ginger when appropriate.
4. Delegate Xactimate CAT/SEL and line-item research to The Professor when
   appropriate.
5. For a full estimate or comprehensive claim review, consult all relevant
   specialists and combine their findings.
6. Do not invent measurements, damage, code requirements, CAT codes, SEL codes,
   units, or pricing.
7. Clearly identify missing information and any assumptions.
8. Return a polished, practical answer suitable for a claims professional.
""",
                expected_output=(
                    "A direct answer to the user's question or a comprehensive "
                    "claim report containing documented observations, code "
                    "considerations, and verified Xactimate recommendations. "
                    "Unknown or unverified information must be clearly labeled."
                ),
                agent=skipper,
            )

            claim_crew = Crew(
                agents=[
                    skipper,
                    gilligan,
                    ginger,
                    professor,
                ],
                tasks=[master_task],
                process=Process.sequential,
                verbose=False,
            )

            crew_output = claim_crew.kickoff()

            final_text = getattr(crew_output, "raw", None) or str(crew_output)

            st.subheader("📋 Final Output")
            st.markdown(final_text)

        except Exception as error:
            st.error("The crew could not complete the request.")
            st.exception(error)
