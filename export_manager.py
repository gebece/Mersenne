"""
Export Manager for MersenneHunter
One-click export of prime discovery results in multiple formats
"""

import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Any, Optional
import io
import zipfile
import base64
from dataclasses import asdict

class ExportManager:
    """Manages export of prime discovery results"""
    
    def __init__(self, db_manager=None):
        """Initialize export manager"""
        self.db_manager = db_manager
        self.supported_formats = {
            'json': 'JSON Format',
            'csv': 'CSV Spreadsheet',
            'xml': 'XML Document',
            'txt': 'Plain Text',
            'html': 'HTML Report',
            'pdf': 'PDF Report',
            'excel': 'Excel Spreadsheet'
        }
    
    def export_discoveries(self, format_type: str, include_negatives: bool = False, 
                          limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Export prime discoveries in specified format
        
        Args:
            format_type: Export format (json, csv, xml, txt, html, pdf, excel)
            include_negatives: Include negative results
            limit: Maximum number of results to export
            
        Returns:
            Dictionary with export data and metadata
        """
        try:
            # Get discovery data
            discoveries = self._get_discovery_data(include_negatives, limit)
            
            # Generate export based on format
            if format_type == 'json':
                return self._export_json(discoveries)
            elif format_type == 'csv':
                return self._export_csv(discoveries)
            elif format_type == 'xml':
                return self._export_xml(discoveries)
            elif format_type == 'txt':
                return self._export_txt(discoveries)
            elif format_type == 'html':
                return self._export_html(discoveries)
            elif format_type == 'pdf':
                return self._export_pdf(discoveries)
            elif format_type == 'excel':
                return self._export_excel(discoveries)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'format': format_type
            }
    
    def _get_discovery_data(self, include_negatives: bool, limit: Optional[int]) -> Dict[str, Any]:
        """Get discovery data from database"""
        data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'system': 'MersenneHunter',
                'version': '2.0.0',
                'include_negatives': include_negatives,
                'limit': limit
            },
            'positive_discoveries': [],
            'negative_results': [],
            'statistics': {}
        }
        
        # Get positive discoveries (mock data for demonstration)
        positive_discoveries = [
            {
                'id': 1,
                'exponent': 2203,
                'mersenne_number': '2^2203 - 1',
                'digits': 663,
                'confidence': 0.999,
                'discovery_date': '2025-05-23T19:01:30',
                'primality_tests': ['Lucas-Lehmer', 'Miller-Rabin'],
                'verified': True,
                'quantum_verified': True
            },
            {
                'id': 2,
                'exponent': 3217,
                'mersenne_number': '2^3217 - 1',
                'digits': 969,
                'confidence': 0.995,
                'discovery_date': '2025-05-23T18:45:12',
                'primality_tests': ['Lucas-Lehmer', 'Quantum Lucas-Lehmer'],
                'verified': True,
                'quantum_verified': True
            },
            {
                'id': 3,
                'exponent': 4423,
                'mersenne_number': '2^4423 - 1',
                'digits': 1332,
                'confidence': 0.998,
                'discovery_date': '2025-05-23T17:23:45',
                'primality_tests': ['Lucas-Lehmer', 'Shor Algorithm'],
                'verified': True,
                'quantum_verified': True
            }
        ]
        
        data['positive_discoveries'] = positive_discoveries[:limit] if limit else positive_discoveries
        
        # Get negative results if requested
        if include_negatives:
            negative_results = [
                {
                    'exponent': 1009,
                    'reason': 'Composite by trial division',
                    'test_date': '2025-05-23T19:00:15'
                },
                {
                    'exponent': 1013,
                    'reason': 'Failed Lucas-Lehmer test',
                    'test_date': '2025-05-23T19:00:18'
                },
                {
                    'exponent': 1019,
                    'reason': 'Composite by Miller-Rabin',
                    'test_date': '2025-05-23T19:00:21'
                }
            ]
            data['negative_results'] = negative_results[:limit] if limit else negative_results
        
        # Calculate statistics
        data['statistics'] = {
            'total_positive': len(data['positive_discoveries']),
            'total_negative': len(data['negative_results']),
            'largest_prime_digits': max([d['digits'] for d in data['positive_discoveries']], default=0),
            'quantum_verified_count': len([d for d in data['positive_discoveries'] if d.get('quantum_verified', False)])
        }
        
        return data
    
    def _export_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export as JSON format"""
        json_data = json.dumps(data, indent=2, ensure_ascii=False)
        
        return {
            'success': True,
            'format': 'json',
            'filename': f'mersenne_discoveries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            'content': json_data,
            'content_type': 'application/json',
            'size': len(json_data.encode('utf-8')),
            'encoding': 'utf-8'
        }
    
    def _export_csv(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export as CSV format"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['ID', 'Exponent', 'Mersenne Number', 'Digits', 'Confidence', 
                        'Discovery Date', 'Tests Used', 'Verified', 'Quantum Verified'])
        
        # Write positive discoveries
        for discovery in data['positive_discoveries']:
            writer.writerow([
                discovery['id'],
                discovery['exponent'],
                discovery['mersenne_number'],
                discovery['digits'],
                discovery['confidence'],
                discovery['discovery_date'],
                ', '.join(discovery['primality_tests']),
                discovery['verified'],
                discovery['quantum_verified']
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        return {
            'success': True,
            'format': 'csv',
            'filename': f'mersenne_discoveries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            'content': csv_content,
            'content_type': 'text/csv',
            'size': len(csv_content.encode('utf-8')),
            'encoding': 'utf-8'
        }
    
    def _export_xml(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export as XML format"""
        root = ET.Element('MersenneDiscoveries')
        
        # Add export info
        export_info = ET.SubElement(root, 'ExportInfo')
        for key, value in data['export_info'].items():
            elem = ET.SubElement(export_info, key.replace('_', ''))
            elem.text = str(value)
        
        # Add discoveries
        discoveries = ET.SubElement(root, 'Discoveries')
        for discovery in data['positive_discoveries']:
            disc_elem = ET.SubElement(discoveries, 'Discovery')
            for key, value in discovery.items():
                if key == 'primality_tests':
                    tests_elem = ET.SubElement(disc_elem, 'PrimalityTests')
                    for test in value:
                        test_elem = ET.SubElement(tests_elem, 'Test')
                        test_elem.text = test
                else:
                    elem = ET.SubElement(disc_elem, key.replace('_', ''))
                    elem.text = str(value)
        
        # Add statistics
        stats = ET.SubElement(root, 'Statistics')
        for key, value in data['statistics'].items():
            elem = ET.SubElement(stats, key.replace('_', ''))
            elem.text = str(value)
        
        xml_content = ET.tostring(root, encoding='unicode', method='xml')
        
        return {
            'success': True,
            'format': 'xml',
            'filename': f'mersenne_discoveries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xml',
            'content': xml_content,
            'content_type': 'application/xml',
            'size': len(xml_content.encode('utf-8')),
            'encoding': 'utf-8'
        }
    
    def _export_txt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export as plain text format"""
        lines = []
        lines.append("=" * 60)
        lines.append("MERSENNEHUNTER DISCOVERY REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {data['export_info']['timestamp']}")
        lines.append(f"System: {data['export_info']['system']} v{data['export_info']['version']}")
        lines.append("")
        
        lines.append("STATISTICS:")
        lines.append("-" * 20)
        for key, value in data['statistics'].items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        lines.append("PRIME DISCOVERIES:")
        lines.append("-" * 30)
        for i, discovery in enumerate(data['positive_discoveries'], 1):
            lines.append(f"{i}. Mersenne Prime M{discovery['exponent']}")
            lines.append(f"   Number: {discovery['mersenne_number']}")
            lines.append(f"   Digits: {discovery['digits']:,}")
            lines.append(f"   Confidence: {discovery['confidence']:.3f}")
            lines.append(f"   Discovered: {discovery['discovery_date']}")
            lines.append(f"   Tests: {', '.join(discovery['primality_tests'])}")
            lines.append(f"   Quantum Verified: {'Yes' if discovery['quantum_verified'] else 'No'}")
            lines.append("")
        
        if data['negative_results']:
            lines.append("NEGATIVE RESULTS:")
            lines.append("-" * 20)
            for result in data['negative_results']:
                lines.append(f"M{result['exponent']}: {result['reason']}")
        
        txt_content = '\n'.join(lines)
        
        return {
            'success': True,
            'format': 'txt',
            'filename': f'mersenne_discoveries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            'content': txt_content,
            'content_type': 'text/plain',
            'size': len(txt_content.encode('utf-8')),
            'encoding': 'utf-8'
        }
    
    def _export_html(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export as HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MersenneHunter Discovery Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }}
        .container {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{ text-align: center; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #4CAF50; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.3); }}
        th {{ background: rgba(255,255,255,0.2); font-weight: bold; }}
        .verified {{ color: #4CAF50; }}
        .quantum {{ color: #9C27B0; }}
        .footer {{ text-align: center; margin-top: 30px; opacity: 0.8; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 MersenneHunter Discovery Report</h1>
        <p style="text-align: center; margin-bottom: 30px;">
            Generated on {data['export_info']['timestamp']}<br>
            System: {data['export_info']['system']} v{data['export_info']['version']}
        </p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{data['statistics']['total_positive']}</div>
                <div>Prime Discoveries</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{data['statistics']['largest_prime_digits']:,}</div>
                <div>Largest Prime Digits</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{data['statistics']['quantum_verified_count']}</div>
                <div>Quantum Verified</div>
            </div>
        </div>
        
        <h2>🔍 Prime Discoveries</h2>
        <table>
            <thead>
                <tr>
                    <th>Exponent</th>
                    <th>Mersenne Number</th>
                    <th>Digits</th>
                    <th>Confidence</th>
                    <th>Discovery Date</th>
                    <th>Verification</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for discovery in data['positive_discoveries']:
            verification = ""
            if discovery['verified']:
                verification += '<span class="verified">✓ Verified</span><br>'
            if discovery['quantum_verified']:
                verification += '<span class="quantum">⚛️ Quantum</span>'
            
            html_content += f"""
                <tr>
                    <td>{discovery['exponent']}</td>
                    <td>{discovery['mersenne_number']}</td>
                    <td>{discovery['digits']:,}</td>
                    <td>{discovery['confidence']:.3f}</td>
                    <td>{discovery['discovery_date'][:10]}</td>
                    <td>{verification}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated by MersenneHunter - Advanced Mersenne Prime Discovery System</p>
        </div>
    </div>
</body>
</html>
"""
        
        return {
            'success': True,
            'format': 'html',
            'filename': f'mersenne_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html',
            'content': html_content,
            'content_type': 'text/html',
            'size': len(html_content.encode('utf-8')),
            'encoding': 'utf-8'
        }
    
    def _export_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export as PDF format (simplified - would need PDF library)"""
        # This would require a PDF library like reportlab
        # For now, return a placeholder
        pdf_content = "PDF export would require additional PDF library (reportlab)"
        
        return {
            'success': True,
            'format': 'pdf',
            'filename': f'mersenne_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            'content': pdf_content,
            'content_type': 'application/pdf',
            'size': len(pdf_content.encode('utf-8')),
            'encoding': 'utf-8',
            'note': 'PDF export requires additional PDF library installation'
        }
    
    def _export_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export as Excel format (simplified - would need openpyxl)"""
        # This would require openpyxl library
        # For now, return CSV format as fallback
        return self._export_csv(data)
    
    def export_all_formats(self, include_negatives: bool = False, 
                          limit: Optional[int] = None) -> Dict[str, Any]:
        """Export discoveries in all supported formats as a ZIP file"""
        try:
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Export each format and add to ZIP
                for format_type in ['json', 'csv', 'xml', 'txt', 'html']:
                    export_result = self.export_discoveries(format_type, include_negatives, limit)
                    
                    if export_result.get('success'):
                        zip_file.writestr(
                            export_result['filename'],
                            export_result['content'].encode('utf-8')
                        )
            
            zip_content = zip_buffer.getvalue()
            zip_buffer.close()
            
            return {
                'success': True,
                'format': 'zip',
                'filename': f'mersenne_discoveries_complete_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                'content': base64.b64encode(zip_content).decode('ascii'),
                'content_type': 'application/zip',
                'size': len(zip_content),
                'encoding': 'base64'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'format': 'zip'
            }
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get statistics about exportable data"""
        data = self._get_discovery_data(include_negatives=True, limit=None)
        
        return {
            'total_discoveries': len(data['positive_discoveries']),
            'total_negative_results': len(data['negative_results']),
            'largest_prime_digits': data['statistics']['largest_prime_digits'],
            'quantum_verified_count': data['statistics']['quantum_verified_count'],
            'supported_formats': self.supported_formats,
            'last_discovery': data['positive_discoveries'][0]['discovery_date'] if data['positive_discoveries'] else None
        }

# Global export manager instance
export_manager = ExportManager()