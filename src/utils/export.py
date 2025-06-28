"""Export functionality for OCR results."""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
except ImportError:
    SimpleDocTemplate = None

from ..core.batch_processor import BatchResult
from .config import get_config

logger = logging.getLogger(__name__)


class ExportManager:
    """Manages export functionality for OCR results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize export manager.
        
        Args:
            config: Export configuration. If None, uses global config.
        """
        self.config = config or get_config().get('export', {})
        self.output_directory = self.config.get('output_directory', 'output')
        self.include_metadata = self.config.get('include_metadata', True)
        
        # Ensure output directory exists
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
    
    def export_results(self, results: List[BatchResult], format_type: str = 'json',
                      filename: Optional[str] = None, summary: Optional[Dict[str, Any]] = None) -> str:
        """Export batch results to specified format.
        
        Args:
            results: List of batch results to export
            format_type: Export format ('json', 'csv', 'xlsx', 'pdf')
            filename: Output filename. If None, generates timestamp-based name.
            summary: Optional summary report to include
            
        Returns:
            Path to exported file
        """
        if not results:
            raise ValueError("No results to export")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ocr_results_{timestamp}.{format_type}"
        
        output_path = os.path.join(self.output_directory, filename)
        
        try:
            if format_type == 'json':
                return self._export_json(results, output_path, summary)
            elif format_type == 'csv':
                return self._export_csv(results, output_path)
            elif format_type == 'xlsx':
                return self._export_xlsx(results, output_path, summary)
            elif format_type == 'pdf':
                return self._export_pdf(results, output_path, summary)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting to {format_type}: {e}")
            raise
    
    def _export_json(self, results: List[BatchResult], output_path: str,
                    summary: Optional[Dict[str, Any]] = None) -> str:
        """Export results to JSON format."""
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results),
                'format': 'json',
                'version': '2.0.0'
            },
            'results': [result.to_dict() for result in results]
        }
        
        if summary and self.include_metadata:
            export_data['summary'] = summary
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results exported to JSON: {output_path}")
        return output_path
    
    def _export_csv(self, results: List[BatchResult], output_path: str) -> str:
        """Export results to CSV format."""
        fieldnames = [
            'image_path', 'expected_text', 'extracted_text', 'corrected_text',
            'similarity', 'processing_time', 'ocr_confidence', 'success',
            'error_count', 'errors'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'image_path': result.image_path,
                    'expected_text': result.expected_text,
                    'extracted_text': result.extracted_text,
                    'corrected_text': result.corrected_text,
                    'similarity': result.similarity,
                    'processing_time': result.processing_time,
                    'ocr_confidence': result.ocr_confidence,
                    'success': result.success,
                    'error_count': len(result.errors),
                    'errors': '; '.join(result.errors)
                }
                writer.writerow(row)
        
        logger.info(f"Results exported to CSV: {output_path}")
        return output_path
    
    def _export_xlsx(self, results: List[BatchResult], output_path: str,
                    summary: Optional[Dict[str, Any]] = None) -> str:
        """Export results to Excel format."""
        if pd is None:
            raise ImportError("pandas is required for Excel export")
        
        # Prepare data for DataFrame
        data = []
        for result in results:
            row = {
                'Image Path': result.image_path,
                'Expected Text': result.expected_text,
                'Extracted Text': result.extracted_text,
                'Corrected Text': result.corrected_text,
                'Similarity': result.similarity,
                'Processing Time (s)': result.processing_time,
                'OCR Confidence': result.ocr_confidence,
                'Success': result.success,
                'Error Count': len(result.errors),
                'Errors': '; '.join(result.errors)
            }
            data.append(row)
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main results sheet
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Summary sheet if provided
            if summary and self.include_metadata:
                summary_data = []
                
                # Summary statistics
                if 'summary' in summary:
                    for key, value in summary['summary'].items():
                        summary_data.append({'Metric': key.replace('_', ' ').title(), 'Value': value})
                
                # Error analysis
                if 'error_analysis' in summary:
                    summary_data.append({'Metric': '', 'Value': ''})  # Empty row
                    summary_data.append({'Metric': 'Error Analysis', 'Value': ''})
                    for error_type, count in summary['error_analysis'].items():
                        summary_data.append({'Metric': error_type.replace('_', ' ').title(), 'Value': count})
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Results exported to Excel: {output_path}")
        return output_path
    
    def _export_pdf(self, results: List[BatchResult], output_path: str,
                   summary: Optional[Dict[str, Any]] = None) -> str:
        """Export results to PDF format."""
        if SimpleDocTemplate is None:
            raise ImportError("reportlab is required for PDF export")
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("OCR Processing Results", title_style))
        story.append(Spacer(1, 12))
        
        # Summary section if provided
        if summary and self.include_metadata:
            story.append(Paragraph("Summary", styles['Heading2']))
            
            if 'summary' in summary:
                summary_data = []
                for key, value in summary['summary'].items():
                    if isinstance(value, float):
                        value = f"{value:.3f}"
                    summary_data.append([key.replace('_', ' ').title(), str(value)])
                
                summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(summary_table)
                story.append(Spacer(1, 20))
        
        # Results section
        story.append(Paragraph("Detailed Results", styles['Heading2']))
        
        # Create results table
        table_data = [['Image', 'Success', 'Similarity', 'Errors']]
        
        for result in results:
            image_name = os.path.basename(result.image_path)
            success = "✓" if result.success else "✗"
            similarity = f"{result.similarity:.3f}" if result.similarity > 0 else "N/A"
            error_count = len(result.errors)
            
            table_data.append([
                image_name,
                success,
                similarity,
                str(error_count)
            ])
        
        results_table = Table(table_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(results_table)
        
        # Add detailed error information for failed results
        failed_results = [r for r in results if not r.success or r.errors]
        if failed_results:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Error Details", styles['Heading2']))
            
            for result in failed_results:
                if result.errors:
                    story.append(Paragraph(f"<b>{os.path.basename(result.image_path)}</b>", styles['Normal']))
                    for error in result.errors:
                        story.append(Paragraph(f"• {error}", styles['Normal']))
                    story.append(Spacer(1, 10))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"Results exported to PDF: {output_path}")
        return output_path
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        formats = ['json', 'csv']
        
        if pd is not None:
            formats.append('xlsx')
        
        if SimpleDocTemplate is not None:
            formats.append('pdf')
        
        return formats
    
    def export_single_result(self, result: BatchResult, format_type: str = 'json',
                           filename: Optional[str] = None) -> str:
        """Export a single result.
        
        Args:
            result: Single batch result to export
            format_type: Export format
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        return self.export_results([result], format_type, filename)
