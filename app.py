import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import PDFSearchTool


# ============================================================
# OPTIONAL HEIC SUPPORT
# ============================================================

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

# Different Google/CrewAI libraries may look for either variable.
os.environ["GEMINI_API_KEY"] = clean_key
os.environ["GOOGLE_API_KEY"] = clean_key


# Direct Gemini client used for uploaded-document extraction.
client = genai.Client(api_key=clean_key)


# CrewAI Gemini model.
#
# max_tokens limits the length of each agent response and helps
# prevent unexpectedly large output charges.
llm = LLM(
    model="gemini/gemini-3.5-flash",
    api_key=clean_key,
    temperature=0.2,
    max_tokens=8000,
)


# ============================================================
# XACTIMATE PDF SEARCH TOOLS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
XACTIMATE_DIR = APP_DIR / "xactimate_pdfs"


def make_pdf_config() -> dict:
    """
    Use a local sentence-transformer model for PDF indexing.

    This avoids:
    - Gemini embedding quota usage
    - Gemini embedding API charges
    - OPENAI_API_KEY requirements

    Gemini is still used by the claims agents.
    """
    return {
        "embedding_model": {
            "provider": "sentence-transformer",
            "config": {
                "model": "all-MiniLM-L6-v2",
            },
        },
        "vectordb": {
            "provider": "chromadb",
            "config": {},
        },
    }


@st.cache_resource(
    show_spinner=(
        "Indexing Xactimate reference PDFs. "
        "The first startup may take several minutes..."
    )
)
def load_xactimate_tools():
    """
    Locate every PDF in xactimate_pdfs and create a separate
    CrewAI search tool for each PDF.
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
            config=make_pdf_config(),
        )

        tools.append(tool)

    return tools


try:
    xactimate_tools = load_xactimate_tools()

except Exception as error:
    error_text = str(error)

    st.error("Could not load the Xactimate reference PDFs.")

    if "sentence" in error_text.lower() and "install" in error_text.lower():
        st.warning(
            "The local sentence-transformers package could not be loaded. "
            "Confirm that sentence-transformers is included in requirements.txt."
        )

    elif "no pdf files" in error_text.lower():
        st.warning(
            "No PDF files were found in the xactimate_pdfs directory."
        )

    elif "does not exist" in error_text.lower():
        st.warning(
            "The xactimate_pdfs directory was not found in the deployed project."
        )

    else:
        st.warning(
            "Check the private Streamlit deployment logs for the technical details."
        )

    print(f"Xactimate PDF initialization error: {error!r}")
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
        "You are an experienced property field adjuster. You evaluate hail, "
        "wind, water, fire, interior, and exterior damage. Clearly distinguish "
        "documented facts from assumptions. Never invent measurements, damage, "
        "test-square results, or policy requirements."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=False,
    max_iter=4,
    max_retry_limit=1,
)


ginger = Agent(
    role="Residential Building Code Specialist (Ginger)",
    goal=(
        "Identify potentially applicable residential building-code provisions "
        "and explain what jurisdictional verification is required."
    ),
    backstory=(
        "You are a building-code specialist familiar with the 2021 and 2024 "
        "IRC. Never claim that a provision is mandatory without identifying "
        "the relevant code edition, jurisdiction, factual trigger, and whether "
        "local adoption must be confirmed. Clearly label anything requiring "
        "confirmation by the authority having jurisdiction."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=False,
    max_iter=4,
    max_retry_limit=1,
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
5. When possible, identify which PDF reference supports a recommendation.
6. Do not provide a price unless the price is present in the available
   reference material.
7. Identify any quantities or measurements that still require verification.

Xactimate master category index:

{XACTIMATE_MASTER_INDEX}
""",
    llm=llm,
    tools=xactimate_tools,
    allow_delegation=False,
    verbose=False,
    max_iter=4,
    max_retry_limit=1,
)


skipper = Agent(
    role="Crew Manager (The Skipper)",
    goal=(
        "Analyze the user's request, coordinate the appropriate specialists, "
        "and return a clear and accurate final response."
    ),
    backstory=(
        "You are the claims crew manager. Gilligan handles field observations, "
        "Ginger handles building-code considerations, and The Professor handles "
        "Xactimate codes and line-item research. For simple questions, delegate "
        "only when necessary. For full claim or estimate requests, consult the "
        "relevant specialists and combine their findings. Never present an "
        "assumption as an established fact."
    ),
    llm=llm,
    allow_delegation=True,
    verbose=False,
    max_iter=4,
    max_retry_limit=1,
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
    type=[
        "pdf",
        "jpg",
        "jpeg",
        "png",
        "webp",
        "heic",
    ],
    accept_multiple_files=True,
)

cause_of_loss = st.selectbox(
    "Cause of Loss",
    [
        "Hail",
        "Wind",
        "Water",
        "Fire",
        "Other",
        "None",
    ],
)

if uploaded_files:
    st.caption(f"{len(uploaded_files)} file(s) selected.")


# ============================================================
# UPLOADED-FILE PROCESSING
# ============================================================

def build_uploaded_file_parts(files):
    """
    Convert uploaded PDFs and images into Gemini input parts.
    """
    prompt_text = """
Analyze the attached property-claim materials. They may include property
photos, handwritten or printed scope notes, and EagleView reports.

Extract only information supported by the uploaded files:

1. Roof pitch, squares, facets, ridges, hips, valleys, eaves, and rakes.
2. Directional slope damage counts and test-square observations.
3. Exterior elevation and collateral damage.
4. Interior damage and affected rooms or materials.
5. Building-code provisions explicitly mentioned in the files.
6. Quantities, measurements, and scope notes relevant to estimating.
7. Any unclear, illegible, or missing information.

If a value cannot be verified, state that it could not be determined.
Do not guess.
"""

    parts = [
        types.Part.from_text(text=prompt_text)
    ]

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

            image.save(
                buffer,
                format="JPEG",
                quality=90,
                optimize=True,
            )

        parts.append(
            types.Part.from_bytes(
                data=buffer.getvalue(),
                mime_type="image/jpeg",
            )
        )

    return parts


def extract_uploaded_file_data(files):
    """
    Send uploaded files to Gemini for multimodal data extraction.
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
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=6000,
        ),
    )

    if not response or not response.text:
        raise RuntimeError(
            "Gemini did not return extracted text for the uploaded files."
        )

    return response.text


# ============================================================
# ERROR HELPERS
# ============================================================

def is_quota_error(error) -> bool:
    error_text = str(error).lower()

    quota_indicators = [
        "429",
        "resource_exhausted",
        "quota exceeded",
        "ratelimit",
        "rate limit",
        "generaterequestsperday",
        "generate_content_free_tier_requests",
    ]

    return any(
        indicator in error_text
        for indicator in quota_indicators
    )


def show_quota_message():
    st.error(
        "The Gemini API usage limit has been reached for this "
        "Google Cloud project."
    )

    st.warning(
        "Please try again after the quota resets, or enable billing for "
        "the Google Cloud project connected to this Gemini API key."
    )

    st.info(
        "Gemini usage limits apply to the Google Cloud project, not just "
        "the individual API key. Creating another key in the same project "
        "will not reset the project's quota."
    )


# ============================================================
# RUN THE CLAIMS CREW
# ============================================================

if st.button("Engage Crew 🚀", type="primary"):
    if not user_input.strip() and not uploaded_files:
        st.warning(
            "Please enter a request or upload at least one document or photo."
        )

        st.stop()

    extracted_data = "No files were uploaded."

    if uploaded_files:
        with st.spinner(
            "Processing documents and images with Gemini..."
        ):
            try:
                extracted_data = extract_uploaded_file_data(
                    uploaded_files
                )

                st.success(
                    "File data extracted successfully."
                )

            except Exception as error:
                if is_quota_error(error):
                    show_quota_message()

                else:
                    st.error(
                        "The uploaded documents or images could not "
                        "be processed."
                    )

                    st.warning(
                        "Check the private Streamlit deployment logs "
                        "for the technical details."
                    )

                print(
                    f"Uploaded-file extraction error: {error!r}"
                )

                st.stop()

    request_text = user_input.strip()

    if not request_text:
        request_text = (
            "Review the uploaded claim materials and summarize "
            "the documented scope."
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

1. Determine whether this is a simple question, a document review, or a
   full claim-estimate request.
2. Delegate field-damage analysis to Gilligan when appropriate.
3. Delegate building-code analysis to Ginger when appropriate.
4. Delegate Xactimate CAT/SEL and line-item research to The Professor
   when appropriate.
5. For a full estimate or comprehensive claim review, consult all
   relevant specialists and combine their findings.
6. Do not invent measurements, damage, code requirements, CAT codes,
   SEL codes, units, or pricing.
7. Clearly identify missing information and assumptions.
8. Keep the answer focused on the user's request.
9. Return a polished, practical answer suitable for a claims professional.
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
                tasks=[
                    master_task,
                ],
                process=Process.sequential,
                verbose=False,
            )

            crew_output = claim_crew.kickoff()

            final_text = (
                getattr(crew_output, "raw", None)
                or str(crew_output)
            )

            st.subheader("📋 Final Output")
            st.markdown(final_text)

        except Exception as error:
            if is_quota_error(error):
                show_quota_message()

            else:
                st.error(
                    "The claims crew could not complete the request."
                )

                st.warning(
                    "Check the private Streamlit deployment logs "
                    "for the technical details."
                )

            # Keep the full technical error in Streamlit Cloud logs.
            # Do not expose the traceback to public website visitors.
            print(
                f"Crew execution error: {error!r}"
            )
