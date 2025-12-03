#!/usr/bin/env python3
"""
Archaeological Site Extraction Pipeline
Extracts archaeological site information from academic papers using LLM
"""

import json
import sys
import os
from pathlib import Path
from openai import OpenAI

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# CONFIG: set your default PDF here
PDF_PATH = "test.pdf"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def load_env():
    """Load environment variables from .env if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Note: python-dotenv not installed. "
              "If you want to load OPENAI_API_KEY from a .env file, run: pip install python-dotenv")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        import pypdf
    except ImportError:
        print("Error: pypdf not installed. Run: pip install pypdf")
        sys.exit(1)
    
    print(f"Reading PDF: {pdf_path}")
    text = ""
    
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        num_pages = len(reader.pages)
        print(f"Found {num_pages} pages")
        
        for i, page in enumerate(reader.pages):
            print(f"Extracting page {i+1}/{num_pages}...", end="\r")
            page_text = page.extract_text() or ""
            text += f"\n--- Page {i+1} ---\n"
            text += page_text
    
    print(f"\nExtracted {len(text)} characters")
    return text

def create_extraction_prompt(paper_text):
    """Create the extraction prompt with the paper text"""
    
    prompt = """# Archaeological Site Information Extraction

    ## Task
    Extract information about archaeological sites from the provided academic paper. Your role is to act as a precise data extractor, NOT an interpreter or estimator.

    ## CRITICAL RULES - READ CAREFULLY

    1. **ONLY extract information explicitly stated in the paper**
      - If coordinates are not given, leave the coordinates field empty
      - If a site name is not mentioned, do not invent one
      - If information is ambiguous or unclear, mark it as such

    2. **NEVER:**
      - Guess or estimate coordinates from place names
      - Invent precision that doesn't exist in the source
      - Convert between coordinate systems unless explicitly shown in the paper
      - Fill in missing information based on general knowledge
      - Assume information from context alone

    3. **ALWAYS:**
      - Preserve the exact format of coordinates as written in the paper
      - Note when information is approximate, estimated, or uncertain
      - Include the specific page or section where information was found
      - Flag when coordinates are withheld or stated as "not disclosed"

    ## Information to Extract

    For each archaeological site mentioned in the paper, extract the following:

    ### Site Identification
    - **site_name**: The exact name(s) used in the paper
    - **site_code**: Any alphanumeric codes or identifiers (e.g., "PA-KU-29")
    - **alternative_names**: Other names mentioned for the same site

    ### Location Information
    - **coordinates_explicit**: ONLY if coordinates are explicitly provided
      - Extract the exact text as written (e.g., "5.23°S, 60.12°W")
      - Note the format: decimal_degrees, DMS, UTM, etc.
      - Note the datum if specified (WGS84, SAD69, etc.)
      - Mark precision level: "exact", "approximate", "rounded"
      
    - **location_description**: Textual descriptions of location
      - Examples: "near the confluence of Xingu and Amazon rivers"
      - "15 km north of modern-day Santarém"
      - "in the Upper Tapajós basin"
      
    - **administrative_location**: Modern political boundaries mentioned
      - Country, state/province, municipality, etc.

    - **location_withheld**: Boolean - true if paper explicitly states location is not disclosed

    ### Temporal Information
    - **dating**: Dates or date ranges mentioned (e.g., "1200-1400 CE", "Late Pre-Columbian")
    - **cultural_period**: Cultural phases or periods named
    - **dating_method**: If specified (radiocarbon, thermoluminescence, etc.)
    - **dating_uncertainty**: Any caveats about dating

    ### Site Characteristics
    - **site_type**: Settlement, mound, ceremonial center, earthwork, etc.
    - **site_features**: Specific features mentioned (e.g., "circular plaza", "defensive ditch")
    - **site_size**: Dimensions if provided (area, diameter, length)
    - **site_condition**: Preservation state if mentioned

    ### Context
    - **study_type**: Is this a site being actively studied or just mentioned for comparison?
    - **source_location**: Page number(s) or section where this information appears
    - **confidence_level**: Your assessment - "high" (explicit coordinates + clear description), "medium" (clear description but no coordinates), "low" (vague or passing mention)
    - **extraction_notes**: Any important caveats, ambiguities, or clarifications

    ## Output Format

    Return a JSON object with this structure:

    {
      "paper_metadata": {
        "title": "extracted from paper",
        "authors": ["list", "of", "authors"],
        "year": "publication year",
        "doi": "if available"
      },
      "extraction_summary": {
        "total_sites_found": 0,
        "sites_with_explicit_coordinates": 0,
        "sites_with_descriptions_only": 0,
        "extraction_date": "YYYY-MM-DD"
      },
      "sites": [
        {
          "site_name": "string or null",
          "site_code": "string or null",
          "alternative_names": [],
          "coordinates": {
            "has_explicit_coordinates": false,
            "raw_text": null,
            "format": null,
            "latitude": null,
            "longitude": null,
            "datum": null,
            "precision_level": null
          },
          "location_description": null,
          "administrative_location": {
            "country": null,
            "state_province": null,
            "other": null
          },
          "location_withheld": false,
          "temporal": {
            "dating": null,
            "cultural_period": null,
            "dating_method": null,
            "uncertainty": null
          },
          "characteristics": {
            "site_type": null,
            "features": [],
            "size": null,
            "condition": null
          },
          "metadata": {
            "study_type": null,
            "source_location": null,
            "confidence_level": null,
            "extraction_notes": null
          }
        }
      ]
    }

    ## Your Task

    Extract all archaeological site information from the following paper:

    """
    
    return prompt + "\n" + paper_text + "\n\nReturn ONLY the JSON output, no additional commentary."

def extract_sites_with_llm(paper_text):
    """Use ChatGPT to extract site information, reading API key from environment."""
    
    # Initialize OpenAI client (uses OPENAI_API_KEY from env)
    client = OpenAI()
    
    print("\nSending to ChatGPT for extraction...")
    
    # Create the prompt
    prompt = create_extraction_prompt(paper_text)
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=16000
    )
    
    # Extract response
    response_text = response.choices[0].message.content
    
    print("Extraction complete!")
    
    # Parse JSON response
    try:
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        extracted_data = json.loads(response_text)
        return extracted_data
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON response: {e}")
        print("Returning raw response text...")
        return {"raw_response": response_text, "parse_error": str(e)}

def save_output(data, output_path):
    """Save extracted data to JSON file"""
    print(f"\nSaving output to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Done!")

def main():
    """Main pipeline function (no CLI arguments)."""

    print("=" * 60)
    print("Archaeological Site Extraction Pipeline")
    print("=" * 60)

    # 1. Load .env (if python-dotenv is installed)
    load_env()

    # 2. Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        print("Create a .env file with:")
        print("  OPENAI_API_KEY=your-key-here")
        print("and/or export it in your shell.")
        sys.exit(1)

    # 3. Check that PDF exists
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        print("Update PDF_PATH at the top of the script.")
        sys.exit(1)

    # 4. Build output file name based on input file name
    #    e.g. "paper.pdf" -> "paper_extracted_sites.json"
    output_path = pdf_path.with_suffix("")  # remove .pdf
    output_path = Path(str(output_path) + "_extracted_sites.json")

    print(f"Input PDF : {pdf_path}")
    print(f"Output JSON: {output_path}")

    # 5. Extract text from PDF
    paper_text = extract_text_from_pdf(pdf_path)

    # 6. Extract sites using LLM
    extracted_data = extract_sites_with_llm(paper_text)

    # 7. Save output
    save_output(extracted_data, output_path)

    # 8. Print summary if available
    if isinstance(extracted_data, dict) and "extraction_summary" in extracted_data:
        summary = extracted_data["extraction_summary"]
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total sites found: {summary.get('total_sites_found', 0)}")
        print(f"Sites with coordinates: {summary.get('sites_with_explicit_coordinates', 0)}")
        print(f"Sites with descriptions only: {summary.get('sites_with_descriptions_only', 0)}")

if __name__ == "__main__":
    main()