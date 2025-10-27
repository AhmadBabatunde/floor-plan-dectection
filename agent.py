#
# final_pdf_annotator_debugged.py
#
# Enhanced version with debugging capabilities and fixes for common annotation issues
#

from typing import List, Optional, Any, Annotated
from dataclasses import dataclass, asdict
import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import cv2
from ultralytics import YOLO
from langchain_core.tools import tool
import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from dotenv import load_dotenv
import json

# Load YOLO model (make sure 'best.pt' exists)
try:
    yolo_model = YOLO('best.pt')
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    yolo_model = None

# --- Setup ---
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# --- Data Structures ---
@dataclass
class Annotation:
    """A dataclass to hold information about a single PDF annotation."""
    type: str  # 'highlight', 'circle', 'rectangle', 'note', 'arrow', 'dot'
    page: int
    coordinates: List[float]
    text: Optional[str] = None
    color: Optional[str] = "red"
    line_width: Optional[float] = 1.5

# --- State for the Graph ---
class GraphState(MessagesState):
    """Represents the state of our workflow."""
    pdf_path: str
    output_path: str

# --- Base Tools ---
@tool
def load_pdf(pdf_path: str) -> str:
    """Load a PDF file and get its total number of pages."""
    try:
        if not os.path.exists(pdf_path):
            return f"Error: PDF file not found at '{pdf_path}'."
        with fitz.open(pdf_path) as pdf_document:
            return f"PDF '{pdf_path}' loaded. Total pages: {len(pdf_document)}"
    except Exception as e:
        return f"Error loading PDF: {str(e)}"

@tool
def get_page_info(pdf_path: str, page_number: int) -> str:
    """Get detailed info for a PDF page, including dimensions and text with coordinates (bbox), to find elements to annotate."""
    try:
        with fitz.open(pdf_path) as pdf_document:
            if not 1 <= page_number <= len(pdf_document):
                return f"Invalid page number. PDF has {len(pdf_document)} pages."
            page = pdf_document.load_page(page_number - 1)
            info = {
                "page_number": page_number,
                "dimensions": {"width": page.rect.width, "height": page.rect.height},
                "text_blocks": [
                    {"text": span["text"].strip(), "bbox": [round(c, 2) for c in span["bbox"]]}
                    for block in page.get_text("dict")["blocks"] if "lines" in block
                    for line in block["lines"]
                    for span in line["spans"] if span["text"].strip()
                ]
            }
            return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error getting page info: {str(e)}"

# --- Annotation Instruction Tools ---
def create_annotation_instruction(ann_type, page, coords, text=None, color=None):
    annotation = Annotation(type=ann_type, page=page, coordinates=coords, text=text, color=color)
    return json.dumps(asdict(annotation))

@tool
def convert_pdf_page_to_image(pdf_path: str, page_number: int, dpi: int = 300) -> str:
    """
    Converts a specific page of a PDF file into a high-resolution PNG image.
    """
    try:
        if not os.path.exists(pdf_path):
            return f"Error: PDF file not found at '{pdf_path}'."
        
        doc = fitz.open(pdf_path)
        if not 1 <= page_number <= len(doc):
            return f"Error: Invalid page number. PDF has {len(doc)} pages."
            
        page = doc.load_page(page_number - 1)
        
        # Define the output image path
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        image_path = f"{base_name}_page_{page_number}.png"
        
        # Render page to an image
        pixmap = page.get_pixmap(dpi=dpi)
        pixmap.save(image_path)
        
        doc.close()
        
        print(f"DEBUG: Image saved to {image_path}")
        return image_path
        
    except Exception as e:
        return f"Error converting PDF page to image: {str(e)}"

@tool
def get_object_dimensions(image_path: str) -> str:
    """
    Identifies objects in an image and returns a list of detected objects
    with their class names and bounding box dimensions.
    """
    try:
        if yolo_model is None:
            return "Error: YOLO model not loaded. Object detection unavailable."
            
        img = cv2.imread(image_path)
        if img is None:
            return f"Error: Could not load image at '{image_path}'."

        results = yolo_model(img)
        detected_objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]

                detected_objects.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 2),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                })

        print(f"DEBUG: Detected {len(detected_objects)} objects")
        return json.dumps(detected_objects, indent=2)

    except Exception as e:
        return f"Error during object detection: {str(e)}"

@tool
def add_highlight_instruction(page_number: int, x1: float, y1: float, x2: float, y2: float) -> str:
    """Records an instruction to highlight an area. Color is yellow."""
    result = create_annotation_instruction('highlight', page_number, [x1, y1, x2, y2], color="yellow")
    print(f"DEBUG: Created highlight instruction: {result}")
    return result

@tool
def add_circle_instruction(page_number: int, center_x: float, center_y: float, radius: float, color: str = "blue") -> str:
    """Records an instruction to encircle an item."""
    coords = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
    result = create_annotation_instruction('circle', page_number, coords, color=color)
    print(f"DEBUG: Created circle instruction: {result}")
    return result

@tool
def add_rectangle_instruction(page_number: int, x1: float, y1: float, x2: float, y2: float, color: str = "red") -> str:
    """Records an instruction to outline an area with a rectangle."""
    result = create_annotation_instruction('rectangle', page_number, [x1, y1, x2, y2], color=color)
    print(f"DEBUG: Created rectangle instruction: {result}")
    return result

@tool
def add_note_instruction(page_number: int, x: float, y: float, label_text: str) -> str:
    """Records an instruction to add a text note or label."""
    result = create_annotation_instruction('note', page_number, [x, y], text=label_text, color="red")
    print(f"DEBUG: Created note instruction: {result}")
    return result

@tool
def add_arrow_instruction(page_number: int, start_x: float, start_y: float, end_x: float, end_y: float) -> str:
    """Records an instruction to add an arrow (callout) pointing from a start point to an end point."""
    result = create_annotation_instruction('arrow', page_number, [start_x, start_y, end_x, end_y], color="red")
    print(f"DEBUG: Created arrow instruction: {result}")
    return result

@tool
def add_count_marker_instruction(page_number: int, x: float, y: float) -> str:
    """Records an instruction to place a small red dot on an item, typically for counting."""
    result = create_annotation_instruction('dot', page_number, [x, y], color="red")
    print(f"DEBUG: Created dot instruction: {result}")
    return result

# --- Final Execution Tool with Enhanced Debugging ---
@tool
def apply_annotations_to_pdf(pdf_path: str, annotations_json: List[str], output_path: str) -> str:
    """Applies a list of annotation instructions to a PDF and saves the result. This is the final step."""
    try:
        print(f"DEBUG: Starting annotation application...")
        print(f"DEBUG: PDF path: {pdf_path}")
        print(f"DEBUG: Output path: {output_path}")
        print(f"DEBUG: Number of annotation instructions: {len(annotations_json)}")
        
        # Parse annotations
        annotations = []
        # Normalize input to a list of dicts
        normalized_items = []
        try:
            if isinstance(annotations_json, str):
                try:
                    parsed = json.loads(annotations_json)
                    if isinstance(parsed, list):
                        normalized_items = parsed
                    elif isinstance(parsed, dict):
                        normalized_items = [parsed]
                    else:
                        print("DEBUG: annotations_json string parsed to unexpected type; attempting line-split parse")
                except Exception:
                    # Try to parse as newline-delimited JSON objects
                    parts = [p.strip().rstrip(',') for p in annotations_json.splitlines() if p.strip()]
                    for p in parts:
                        try:
                            normalized_items.append(json.loads(p))
                        except Exception as e:
                            print(f"DEBUG: Skipping non-JSON line during parse: {p[:80]}... Error: {e}")
            elif isinstance(annotations_json, list):
                normalized_items = annotations_json
            else:
                print(f"DEBUG: annotations_json had unexpected type: {type(annotations_json)}")
        except Exception as e:
            print(f"DEBUG: Error normalizing annotations_json: {e}")

        for i, item in enumerate(normalized_items):
            try:
                if isinstance(item, str):
                    ann_dict = json.loads(item)
                elif isinstance(item, dict):
                    ann_dict = item
                else:
                    print(f"DEBUG: Skipping item {i+1} due to unexpected type: {type(item)}")
                    continue
                annotation = Annotation(**ann_dict)
                annotations.append(annotation)
                print(f"DEBUG: Parsed annotation {i+1}: {annotation}")
            except Exception as e:
                print(f"DEBUG: Failed to parse annotation {i+1}: {item}, Error: {e}")
                continue
        
        if not annotations:
            return "Error: No valid annotations to apply!"
        
        # Open PDF and apply annotations
        with fitz.open(pdf_path) as pdf_document:
            print(f"DEBUG: PDF has {len(pdf_document)} pages")
            
            for i, ann in enumerate(annotations):
                print(f"DEBUG: Processing annotation {i+1}: {ann.type} on page {ann.page}")
                
                if not 1 <= ann.page <= len(pdf_document):
                    print(f"DEBUG: Skipping annotation - invalid page {ann.page}")
                    continue
                    
                page = pdf_document.load_page(ann.page - 1)
                
                # Get color - with fallback
                try:
                    if ann.color in ['red', 'blue', 'green', 'yellow', 'black', 'white']:
                        color = fitz.utils.getColor(ann.color)
                    else:
                        color = (1, 0, 0)  # Default to red
                except:
                    color = (1, 0, 0)  # Default to red
                
                print(f"DEBUG: Using color: {color}")
                
                # Apply annotation based on type
                if ann.type == "highlight":
                    rect = fitz.Rect(ann.coordinates)
                    print(f"DEBUG: Creating highlight at {rect}")
                    # Try to highlight actual text within the rectangle; fallback to a semi-transparent filled rectangle
                    try:
                        words = page.get_text("words") or []
                        quads = [fitz.Quad(fitz.Rect(w[:4])) for w in words if fitz.Rect(w[:4]).intersects(rect)]
                    except Exception:
                        quads = []
                    if quads:
                        annot = page.add_highlight_annot(quads)
                        annot.set_colors(stroke=color, fill=color)
                        annot.set_opacity(0.35)
                    else:
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=color, fill=color)
                        annot.set_border(width=ann.line_width or 1.5)
                        annot.set_opacity(0.25)
                    try:
                        annot.set_flags(getattr(fitz, "ANNOT_FLAG_PRINT", 4))
                    except Exception:
                        pass
                    annot.update()
                
                elif ann.type == "rectangle":
                    rect = fitz.Rect(ann.coordinates)
                    print(f"DEBUG: Creating rectangle at {rect}")
                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=color)
                    annot.set_border(width=ann.line_width or 1.5)
                    try:
                        annot.set_flags(getattr(fitz, "ANNOT_FLAG_PRINT", 4))
                    except Exception:
                        pass
                    annot.update()
                
                elif ann.type == "circle":
                    rect = fitz.Rect(ann.coordinates)
                    print(f"DEBUG: Creating circle at {rect}")
                    annot = page.add_circle_annot(rect)
                    annot.set_colors(stroke=color)
                    annot.set_border(width=ann.line_width or 1.5)
                    try:
                        annot.set_flags(getattr(fitz, "ANNOT_FLAG_PRINT", 4))
                    except Exception:
                        pass
                    annot.update()
                
                elif ann.type == "note":
                    point = fitz.Point(ann.coordinates[0], ann.coordinates[1])
                    text_str = ann.text or "Note"
                    # Use a FreeText annotation so it reliably appears in viewers and print/export
                    rect = fitz.Rect(point.x, point.y, point.x + 180, point.y + 40)
                    print(f"DEBUG: Adding free text '{text_str}' at {rect}")
                    annot = page.add_freetext_annot(rect, text_str)
                    annot.set_colors(stroke=color, fill=(1, 1, 1))
                    annot.set_border(width=ann.line_width or 1.0)
                    try:
                        annot.set_flags(getattr(fitz, "ANNOT_FLAG_PRINT", 4))
                    except Exception:
                        pass
                    annot.update()
                
                elif ann.type == "arrow":
                    start_point = fitz.Point(ann.coordinates[0], ann.coordinates[1])
                    end_point = fitz.Point(ann.coordinates[2], ann.coordinates[3])
                    print(f"DEBUG: Creating arrow from {start_point} to {end_point}")
                    annot = page.add_line_annot(start_point, end_point)
                    annot.set_line_ends(0, 5)  # No start, closed arrow end
                    annot.set_colors(stroke=color)
                    annot.set_border(width=ann.line_width or 2.0)
                    try:
                        annot.set_flags(getattr(fitz, "ANNOT_FLAG_PRINT", 4))
                    except Exception:
                        pass
                    annot.update()
                
                elif ann.type == "dot":
                    radius = 5
                    center = fitz.Point(ann.coordinates[0], ann.coordinates[1])
                    rect = fitz.Rect(center.x - radius, center.y - radius, center.x + radius, center.y + radius)
                    print(f"DEBUG: Creating dot at {center}")
                    annot = page.add_circle_annot(rect)
                    annot.set_colors(stroke=color, fill=color)
                    try:
                        annot.set_flags(getattr(fitz, "ANNOT_FLAG_PRINT", 4))
                    except Exception:
                        pass
                    annot.update()
                
                else:
                    print(f"DEBUG: Unknown annotation type: {ann.type}")

            # Save the PDF
            # Finalize: ensure annotations will appear in all viewers/printed exports by flattening them
            try:
                for page_index in range(len(pdf_document)):
                    page_obj = pdf_document.load_page(page_index)
                    annot = page_obj.first_annot
                    while annot is not None:
                        try:
                            # ensure print flag, then flatten
                            annot.set_flags(getattr(fitz, "ANNOT_FLAG_PRINT", 4))
                            annot.update()
                            next_annot = annot.next
                            annot.flatten()
                            annot = next_annot
                        except Exception:
                            annot = getattr(annot, 'next', None)
            except Exception as e:
                print(f"DEBUG: Failed to flatten annotations: {e}")

            print(f"DEBUG: Saving PDF to {output_path}")
            pdf_document.save(output_path, garbage=4, deflate=True, clean=True, incremental=False)
            
        # Verify the file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"DEBUG: Output file created successfully, size: {file_size} bytes")
            return f"Success! Applied {len(annotations)} annotations. Annotated PDF saved to: {output_path}"
        else:
            return f"Error: Output file was not created at {output_path}"
            
    except Exception as e:
        print(f"DEBUG: Exception in apply_annotations_to_pdf: {e}")
        import traceback
        traceback.print_exc()
        return f"Error applying annotations: {str(e)}"

# --- Agent Definition ---
all_tools = [
    load_pdf, get_page_info, add_highlight_instruction, add_circle_instruction,
    add_rectangle_instruction, add_note_instruction, add_arrow_instruction,
    add_count_marker_instruction, apply_annotations_to_pdf, get_object_dimensions,
    convert_pdf_page_to_image
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI assistant for annotating architectural blueprints in PDF format.
Your workflow is as follows:

1.  **Load PDF First**: Always start by calling `load_pdf` to verify the PDF exists and get page count.

2.  **Preparation (Image Conversion)**: If the task requires visual analysis (e.g., identifying specific objects like doors, windows, or equipment), convert the relevant PDF page into an image using the `convert_pdf_page_to_image` tool. You will need to specify the `pdf_path` and `page_number`.

3.  **Visual Analysis (Object Detection)**: After successfully converting the page to an image, use the `get_object_dimensions` tool with the path to the newly created image. This tool will identify key objects and their bounding box coordinates.

4.  **Textual Analysis (Information Extraction)**: Use the `get_page_info` tool to extract text and its coordinates from the PDF page. This provides textual context and helps in locating specific labels or descriptions.

5.  **Create Annotation Instructions**: Based on the user's request, call the appropriate annotation instruction tool for each item you want to mark. COLLECT ALL THE JSON RESPONSES from these tools - you will need to pass them to the final step.
    - To **highlight**: Use `add_highlight_instruction`.
    - To **encircle**: Use `add_circle_instruction`.
    - To **outline**: Use `add_rectangle_instruction`.
    - For **callouts/arrows**: Use `add_arrow_instruction`.
    - To **add text labels**: Use `add_note_instruction`.
    - To **count items**: Use `add_count_marker_instruction` to place a dot on each item.

6.  **Final Application**: After you have created ALL necessary annotation instructions, make *one final call* to the `apply_annotations_to_pdf` tool. Pass the complete list of JSON strings you received from step 5 to this tool.

IMPORTANT: Your final action MUST be to call `apply_annotations_to_pdf` with ALL the JSON annotation instructions you collected. Do not skip this step!
"""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=openai_api_key)
agent = create_tool_calling_agent(llm, all_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)

# --- Graph Definition ---
def should_continue(state: GraphState) -> str:
    return "action" if state["messages"][-1].tool_calls else END

def call_agent(state: GraphState):
    response = agent_executor.invoke(state)
    return {"messages": [AIMessage(content=response["output"])]}

workflow = StateGraph(GraphState)
workflow.add_node("agent", call_agent)
workflow.add_node("action", ToolNode(all_tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
workflow.add_edge("action", "agent")
graph = workflow.compile()

# --- Main Execution Logic ---
def process_pdf_with_annotations(pdf_path: str, user_request: str, output_path: str = None):
    """Process PDF with annotations and provide detailed debugging output."""
    
    print(f"DEBUG: Starting process_pdf_with_annotations")
    print(f"DEBUG: PDF path: {pdf_path}")
    print(f"DEBUG: User request: {user_request}")
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found at {pdf_path}")
        return

    if output_path is None:
        base, ext = os.path.splitext(pdf_path)
        output_path = f"{base}_annotated{ext}"
    
    print(f"DEBUG: Output path: {output_path}")

    initial_state = {
        "messages": [HumanMessage(content=f"Task: {user_request}. The input PDF is '{pdf_path}'. Save the final PDF to '{output_path}'.")],
        "pdf_path": pdf_path,
        "output_path": output_path,
    }
    
    print(f"DEBUG: Invoking graph...")
    final_state = graph.invoke(initial_state, {"recursion_limit": 25})
    
    print("\n--- Processing Complete ---")
    final_message = final_state['messages'][-1].content
    print(f"Final Message: {final_message}")
    
    # Check if output file exists
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"✅ Annotated PDF is available at: {output_path} (Size: {file_size} bytes)")
        
        # Quick verification - try to open and count annotations
        try:
            with fitz.open(output_path) as doc:
                total_annots = 0
                for page in doc:
                    try:
                        for _ in (page.annots() or []):
                            total_annots += 1
                    except Exception:
                        annot = page.first_annot
                        while annot is not None:
                            total_annots += 1
                            annot = annot.next
                print(f"✅ PDF contains {total_annots} annotations")
        except Exception as e:
            print(f"⚠️ Could not verify annotations: {e}")
    else:
        print(f"❌ Output file not found at: {output_path}")
    return output_path if os.path.exists(output_path) else None

# --- Example Usage ---
if __name__ == "__main__":
    # Test with simple annotation first
    pdf_file = "test_pdf.pdf"
    
    # Simple test request
    simple_request = "Add a red rectangle around any text on page 1"
    print(f"--- Running Simple Test: {simple_request} ---")
    process_pdf_with_annotations(pdf_file, simple_request, "test_simple_annotated.pdf")
    
    # Your original request
    request_2 = "On the pdf, circle the stair case, Count all the wall and place the number on them"
    print(f"\n--- Running Original Request: {request_2} ---")
    process_pdf_with_annotations(pdf_file, request_2, "test_annotated.pdf")