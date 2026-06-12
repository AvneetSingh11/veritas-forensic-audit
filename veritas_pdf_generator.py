from fpdf import FPDF
from datetime import datetime
import hashlib
import os

class VeritasAuditReport(FPDF):
    def header(self):
        # Professional Header
        self.set_font("helvetica", "B", 18)
        self.set_text_color(180, 0, 0) # Threat-level red accent
        self.cell(0, 10, "VERITAS MEDIA FORENSICS", border=False, ln=True, align="L")
        
        self.set_font("helvetica", "I", 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Certified Synthetic Audio Audit", border=False, ln=True, align="L")
        
        self.set_draw_color(150, 150, 150)
        self.line(10, 28, 200, 28)
        self.ln(10)

    def footer(self):
        # Legally required footer
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Confidential Audit Report | Page {self.page_no()} | Veritas Forensics 2026", align="C")

def generate_client_pdf(file_path, audit_results, output_path="Veritas_Final_Report.pdf"):
    """
    Takes the output from your C2PA/AI pipeline and generates a corporate PDF.
    """
    pdf = VeritasAuditReport()
    pdf.add_page()
    
    # 1. Generate a cryptographic hash of the file so the client can prove 
    # they are looking at the exact file you audited.
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()

    # 2. File Metadata Section
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Asset Metadata & Custody", ln=True)
    
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Analyzed File: {os.path.basename(file_path)}", ln=True)
    pdf.cell(0, 6, f"Timestamp of Audit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST", ln=True)
    pdf.cell(0, 6, f"SHA-256 Checksum: {file_hash}", ln=True)
    pdf.ln(5)

    # 3. Provenance & Cryptography (C2PA)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "2. Cryptographic Provenance (C2PA/SynthID)", ln=True)
    pdf.set_font("helvetica", "", 10)
    
    # If the router found a manifest, highlight it in red. If missing, note it.
    origin = audit_results.get("origin_data", "Unknown")
    pdf.cell(0, 6, f"Cryptographic Origin: {origin}", ln=True)
    pdf.ln(5)

    # 4. Final Verdict & AI Scoring
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "3. Acoustic Forensic Verdict", ln=True)
    
    verdict = audit_results.get("verdict", "INCONCLUSIVE")
    confidence = audit_results.get("confidence", 0.0)
    mechanism = audit_results.get("certainty_mechanism", "N/A")
    
    # Big, bold verdict text
    pdf.set_font("helvetica", "B", 16)
    if "SYNTHETIC" in verdict:
        pdf.set_text_color(200, 0, 0) # Red for fake
    else:
        pdf.set_text_color(0, 150, 0) # Green for real
        
    pdf.cell(0, 10, f"VERDICT: {verdict} ({confidence}%)", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Primary Detection Mechanism: {mechanism}", ln=True)
    pdf.ln(10)

    # 5. The Liability Disclaimer (Crucial for B2B)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "4. Limitation of Liability", ln=True)
    pdf.set_font("helvetica", "", 8)
    disclaimer = (
        "This audit relies on probabilistic machine learning models and cryptographic extraction. "
        "While engineered to state-of-the-art accuracy, artificial intelligence detection is not absolute. "
        "This document is provided for advisory risk-mitigation purposes only and does not constitute a "
        "definitive legal ruling on the authenticity of the media."
    )
    pdf.multi_cell(0, 5, disclaimer)

    # Export the PDF
    pdf.output(output_path)
    print(f"Generated Enterprise Report: {output_path}")
