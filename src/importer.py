"""
RWTH Transcript Importer

This module provides functionality to import and parse RWTH Aachen University transcript PDFs.
It extracts module information including grades, ECTS credits, and semester data for grade analysis.
"""

import pandas as pd
import PyPDF2
import pdfplumber
import re
import json
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# Regex helpers usable across the parser
# ------------------------------------------------------------------
FLOAT_RE = r'\d{1,3}[,.]\d{1,2}'            # 4,00  •  12.5  •  1,7
DATE_RE  = r'(?:\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})'

SEM_RE   = r'\b\d{2}[WS]\b'                 # 22W, 23S  (WS = Winter‑, SS = Sommer‑Semester)

# Any credit value above this is considered an aggregate area, not an individual module
LARGE_CREDITS_THRESHOLD = 20  

# Phrases that mark summary / legend rows we must ignore
SUMMARY_KEYWORDS_RE = re.compile(r'\b(overall|gesamt(?:note|credits)?|grades?:)', re.I)

# ------------------------------------------------------------------
# Helper to convert semester codes to sortable keys
# ------------------------------------------------------------------
def _semester_sort_key(code: str) -> Tuple[int, int]:
    """
    Convert a semester code like '24W' or '23S' into a sortable tuple.
    Earlier semesters → smaller tuples.
    Spring/Summer ('S') comes *before* Winter ('W') of the same year.
    """
    if not code or len(code) < 3:
        return (9999, 9)
    try:
        year = int(code[:2])
        term = 0 if code[-1].upper() == 'S' else 1  # S < W
        return (year, term)
    except ValueError:
        return (9999, 9)

def _grab_cp(tokens: List[str], grade_idx: int) -> Optional[float]:
    """
    Return the first numeric token *after* the grade token that looks like
    a credit‑point value (CP/ECTS).  Falls back to None.
    """
    for tok in tokens[grade_idx + 1:]:
        if re.fullmatch(FLOAT_RE, tok):
            return float(tok.replace(',', '.'))
    return None


def load_sample_data() -> pd.DataFrame:
    """
    Load sample module data from JSON file.
    
    Returns:
        pd.DataFrame: DataFrame containing sample module data with proper semester handling
    """
    try:
        data_path = Path(__file__).parent.parent / "data" / "sample_modules.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Import here to avoid circular import
        from .analysis import prepare_module_dataframe
        return prepare_module_dataframe(data['modules'])
    except Exception as e:
        # Fallback to empty DataFrame if file can't be loaded
        print(f"Warning: Could not load sample data: {e}")
        return pd.DataFrame()


class RWTHTranscriptImporter:
    """
    A class to import and parse RWTH Aachen University transcript PDFs.
    
    This class can extract module information including:
    - Module names
    - Grades
    - ECTS credits
    - Semester information
    - Module codes
    """
    
    def __init__(self, pdf_path: Union[str, Path]):
        """
        Initialize the transcript importer.
        
        Args:
            pdf_path: Path to the PDF transcript file
        """
        self.pdf_path = Path(pdf_path)
        self.raw_text = ""
        self.modules = []
        self.parsed_data = {}
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
    
    def extract_text(self) -> str:
        """
        Extract text from the PDF file using multiple methods for better accuracy.
        
        Returns:
            str: Extracted text from the PDF
        """
        text_content = ""
        
        try:
            # Method 1: Using pdfplumber (better for tables and structured text)
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
            
            # Method 2: Fallback to PyPDF2
            try:
                with open(self.pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n"
            except Exception as e2:
                print(f"PyPDF2 extraction also failed: {e2}")
                raise Exception("Could not extract text from PDF using any method")
        
        self.raw_text = text_content
        return text_content
    
    def parse_modules(self) -> List[Dict]:
        """
        Parse module information from the extracted text.
        
        Returns:
            List[Dict]: List of module dictionaries with keys:
                - Module: Name of the module
                - Grade: Grade (float)
                - Credits: ECTS credits (float)
                - Semester: Semester number (int)
                - Courses: List of individual courses within the module
                - module_code: Module code (str, optional)
                - status: Pass/Fail status (str)
        """
        if not self.raw_text:
            self.extract_text()
        
        modules = []
        lines = self.raw_text.split('\n')
        # -------------------------------------------------------------
        # Process only the English sections: any text BETWEEN a line
        # that contains “Certification Examinations” (start flag) and
        # the next line that contains “Notenspiegel” (stop flag).
        # -------------------------------------------------------------
        inside_eng_section = False
        # Track current module context
        current_module: Optional[Dict] = None
        potential_header_line: Optional[str] = None
        potential_header_idx: int = -1

        for idx, raw in enumerate(lines):
            line = raw.strip()
            lower_line = line.lower()

            # Handle section flags
            if "certification examinations" in lower_line:
                inside_eng_section = True
                continue
            if "notenspiegel" in lower_line:
                inside_eng_section = False
                continue
            if not inside_eng_section or not line:
                continue

            grade_matches = re.findall(FLOAT_RE, line)
            is_course = bool(re.search(r'\bBE\b', line))

            if is_course:
                # -----------------------------------------------------------------
                # This is a course line.  Ensure we have a current module header.
                # -----------------------------------------------------------------
                if current_module is None:
                    if potential_header_line:
                        current_module = self._parse_module_line(
                            potential_header_line, potential_header_idx, lines
                        )
                        if current_module:
                            current_module["Courses"] = []
                # If we still failed to obtain a valid module header, skip this course line
                if current_module is None:
                    continue

                course_info = self._parse_course_line(line, idx, lines)
                if course_info:
                    current_module["Courses"].append(course_info)
            else:
                # A non‑course line that *might* be the module header
                if grade_matches or re.search(rf'({FLOAT_RE})\s*(?:ECTS|CP|Credits?)', line, re.I):
                    # If we are already inside a module (and just finished its courses),
                    # finalize it before capturing the next potential header.
                    if current_module and current_module.get("Courses"):
                        self._finalize_module(current_module)
                        modules.append(current_module)
                        current_module = None
                    # Capture this line as the *potential* module header.
                    potential_header_line = line
                    potential_header_idx = idx
                # else: ignore purely textual lines

        # Add the last module if it was still open
        if current_module and current_module.get("Courses"):
            self._finalize_module(current_module)
            modules.append(current_module)
        
        # If no modules found with the heuristic approach, try table-based extraction
        if not modules:
            modules = self._extract_table_data()

        # Remove duplicates caused by the German + English sections
        modules = self._deduplicate_modules(modules)

        # -------------------------------------------------------------
        # Normalise semester numbers based on actual codes (22S, 23W…)
        # -------------------------------------------------------------
        self._renumber_semesters(modules)

        self.modules = modules
        return modules
    
    def _deduplicate_modules(self, modules: List[Dict]) -> List[Dict]:
        """
        Keep only one entry when German + English duplicates occur,
        preferring the *later* (usually English) occurrence.
        The decision key is  (module_code OR normalized module name, grade, credits).
        """
        def _safe_round(val, nd: int = 2):
            try:
                return round(float(val), nd)
            except (TypeError, ValueError):
                return -1

        dedup: Dict[Tuple, Dict] = {}
        for m in modules:
            key = (
                m.get("module_code") or str(m.get("Module", "")).lower(),
                _safe_round(m.get("Grade")),
                _safe_round(m.get("Credits")),
            )
            # Later rows overwrite earlier ones ⇒ English section wins
            dedup[key] = m

        # Preserve original document order of the *last* occurrences
        unique_ordered: List[Dict] = []
        for m in modules:
            key = (
                m.get("module_code") or str(m.get("Module", "")).lower(),
                _safe_round(m.get("Grade")),
                _safe_round(m.get("Credits")),
            )
            if dedup.get(key) is m:
                unique_ordered.append(m)
        return unique_ordered

    def _renumber_semesters(self, modules: List[Dict]) -> None:
        """
        Replace the provisional 'Semester' integers with a 1‑based index
        derived from chronological ordering of semester_code strings
        (YY[S/W]). Spring/Summer ('S') precedes Winter ('W') in the same year.
        """
        # Collect every distinct semester_code
        codes: List[str] = [
            c["semester_code"]
            for m in modules
            for c in m.get("Courses", [])
            if c.get("semester_code")
        ]
        if not codes:
            return  # nothing to normalise

        unique_sorted = sorted(set(codes), key=_semester_sort_key)
        idx_map = {code: i + 1 for i, code in enumerate(unique_sorted)}

        # Update courses and modules
        for m in modules:
            course_sem_numbers: List[int] = []
            for c in m.get("Courses", []):
                code = c.get("semester_code")
                if code in idx_map:
                    num = idx_map[code]
                    c["Semester"] = num
                    course_sem_numbers.append(num)
            if course_sem_numbers:
                m["Semester"] = min(course_sem_numbers)

    def _finalize_module(self, module: Dict) -> None:
        """
        Finalize module data by ensuring proper structure and calculating derived values.
        
        Args:
            module: Module dictionary to finalize
        """
        # If no courses were found, create a single course representing the module
        if not module.get('Courses'):
            module['Courses'] = [{
                'Course': module['Module'],
                'Grade': module['Grade'],
                'Credits': module['Credits'],
                'Semester': module['Semester']
            }]
        
        # Calculate module grade from courses if needed
        if len(module['Courses']) > 1:
            # Recalculate module grade as weighted average of courses
            total_points = sum(
                course['Grade'] * course['Credits']
                for course in module['Courses']
                if (course.get('Grade') is not None and course.get('Credits'))
            )
            total_credits = sum(
                course['Credits'] or 0
                for course in module['Courses']
                if course.get('Credits') is not None
            )
            
            if total_credits > 0:
                module['Grade'] = total_points / total_credits
                module['Credits'] = total_credits
        
    
    def _parse_module_line(self, line: str, line_index: int, all_lines: List[str]) -> Optional[Dict]:
        """
        Parse a single line that appears to contain module information.
        
        Args:
            line: The line to parse
            line_index: Index of the line in the document
            all_lines: All lines from the document for context
            
        Returns:
            Dict or None: Module information dictionary in the expected format
        """
        # -----------------------------------------------------------
        # Grade  (first standalone float token in the line)
        # -----------------------------------------------------------
        tokens = line.split()
        grade_token = None
        grade_idx   = -1
        for idx, tok in enumerate(tokens):
            if re.fullmatch(FLOAT_RE, tok):
                grade_token = tok
                grade_idx   = idx
                break
        # grade is optional for module headers like "Individual Modules N 8.50 …"
        if grade_token is not None:
            grade = float(grade_token.replace(',', '.'))
            # Filter obvious summary rows
            if grade < 0.7 or grade > 5.0:
                grade = None  # treat as missing
        else:
            grade = None
            grade_idx = -1

        # -----------------------------------------------------------
        # Credits  (either labelled or the next float after grade)
        # -----------------------------------------------------------
        ects_match = re.search(rf'({FLOAT_RE})\s*(?:ECTS|CP|Credits?|Leistungspunkte)', line, re.I)
        ects = (
            float(ects_match.group(1).replace(',', '.'))
            if ects_match else _grab_cp(tokens, grade_idx)
        )

        # -----------------------------------------------------------------
        # Filter out summary lines and aggregate study‑area headers
        # -----------------------------------------------------------------
        if (ects and ects > LARGE_CREDITS_THRESHOLD) or SUMMARY_KEYWORDS_RE.search(line):
            return None

        # Extract module code
        module_code_pattern = r'\b([A-Z]{2,4}\d{3,6})\b'
        module_code_matches = re.findall(module_code_pattern, line)

        # Try to extract module name (usually the longest text portion)
        parts = line.split()
        potential_names = []
        for part in parts:
            if (not re.match(FLOAT_RE, part) and
                not re.match(r'\d+[,.]?\d*', part) and
                not part.upper() in ['ECTS', 'CP', 'CREDITS', 'BESTANDEN', 'PASSED'] and
                len(part) > 2):
                potential_names.append(part)

        module_name = ' '.join(potential_names) if potential_names else f"Module_{line_index}"

        module_code = module_code_matches[0] if module_code_matches else None

        # Determine status
        status = None
        if grade is not None:
            status = "bestanden" if grade <= 4.0 else "nicht bestanden"

        # Estimate semester (can be refined based on additional PDF context)
        semester = self._estimate_semester(line_index, all_lines)

        return {
            'Module': module_name,
            'Grade': grade,
            'Credits': ects,
            'Semester': semester,
            'module_code': module_code,
            'status': status,
            'raw_line': line
        }
    
    def _parse_course_line(self, line: str, line_index: int, all_lines: List[str]) -> Optional[Dict]:
        """
        Parse a single line that appears to contain course information.
        
        Args:
            line: The line to parse
            line_index: Index of the line in the document
            all_lines: All lines from the document for context
            
        Returns:
            Dict or None: Course information dictionary in the expected format
        """
        # -----------------------------------------------------------
        # Grade  (first standalone float token in the line)
        # -----------------------------------------------------------
        tokens = line.split()
        grade_token = None
        grade_idx   = -1
        for idx, tok in enumerate(tokens):
            if re.fullmatch(FLOAT_RE, tok):
                grade_token = tok
                grade_idx   = idx
                break
        if grade_token is None:
            return None  # No standalone grade found
        
        grade = float(grade_token.replace(',', '.'))
        # Discard lines whose 'grade' token is clearly not a valid RWTH grade
        # (valid grades lie between 0.7 and 5.0).  Anything outside that range
        # is probably "Overall Credits", "210", etc.
        if grade < 0.7 or grade > 5.0:
            return None

        # -----------------------------------------------------------
        # Credits  (either labelled or the next float after grade)
        # -----------------------------------------------------------
        ects_match = re.search(rf'({FLOAT_RE})\s*(?:ECTS|CP|Credits?|Leistungspunkte)', line, re.I)
        ects = (
            float(ects_match.group(1).replace(',', '.'))
            if ects_match else _grab_cp(tokens, grade_idx)
        )

        # -----------------------------------------------------------------
        # Filter out summary lines and aggregate study‑area headers
        # -----------------------------------------------------------------
        if (ects and ects > LARGE_CREDITS_THRESHOLD) or SUMMARY_KEYWORDS_RE.search(line):
            return None

        # Extract module code
        module_code_pattern = r'\b([A-Z]{2,4}\d{3,6})\b'
        module_code_matches = re.findall(module_code_pattern, line)

        # Try to extract course name (usually the longest text portion)
        parts = line.split()
        potential_names = []
        for part in parts:
            if (not re.match(FLOAT_RE, part) and
                not re.match(r'\d+[,.]?\d*', part) and
                not part.upper() in ['ECTS', 'CP', 'CREDITS', 'BESTANDEN', 'PASSED', 'BE', 'N', 'J', 'T'] and
                len(part) > 2):
                potential_names.append(part)

        course_name = ' '.join(potential_names) if potential_names else f"Course_{line_index}"

        module_code = module_code_matches[0] if module_code_matches else None

        # Determine status
        status = "bestanden" if grade and grade <= 4.0 else "nicht bestanden"

        # Estimate semester (can be refined based on additional PDF context)
        semester = self._estimate_semester(line_index, all_lines)

        # Capture raw semester code if present (e.g. 22W, 23S)
        sem_code_match = re.search(SEM_RE, line)
        if sem_code_match:
            semester_code = sem_code_match.group(0)
        else:
            semester_code = None

        return {
            'Course': course_name,
            'Grade': grade,
            'Credits': ects,
            'Semester': semester,
            'module_code': module_code,
            'status': status,
            'raw_line': line,
            'semester_code': semester_code,
        }
    
    def _estimate_semester(self, line_index: int, all_lines: List[str]) -> int:
        """
        Estimate semester based on position in document and context clues.
        
        Args:
            line_index: Current line index
            all_lines: All lines from the document
            
        Returns:
            int: Estimated semester number
        """
        # Look for semester indicators in surrounding lines
        context_start = max(0, line_index - 10)
        context_end = min(len(all_lines), line_index + 10)
        
        for i in range(context_start, context_end):
            line = all_lines[i].lower()
            # Look for semester patterns
            semester_match = re.search(r'semester\s*(\d+)|(\d+)\.\s*semester|ws\s*(\d{4})|ss\s*(\d{4})', line)
            if semester_match:
                # Extract semester number from different patterns
                groups = semester_match.groups()
                for group in groups:
                    if group and group.isdigit():
                        if len(group) == 4:  # Year format (WS2023, SS2024)
                            # Convert year to approximate semester
                            year = int(group)
                            if 'ws' in line:  # Winter semester
                                return ((year - 2020) * 2) + 1  # Rough estimation
                            else:  # Summer semester
                                return ((year - 2020) * 2) + 2
                        else:
                            return int(group)
        
        # Fallback: estimate based on position in document
        # Assume document is roughly chronological
        relative_position = line_index / len(all_lines)
        estimated_semester = max(1, int(relative_position * 8) + 1)  # Assume max 8 semesters
        return min(estimated_semester, 8)
    
    def _extract_table_data(self) -> List[Dict]:
        """
        Extract data using table extraction methods.
        
        Returns:
            List[Dict]: List of module dictionaries
        """
        modules = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:  # Skip empty tables
                            modules.extend(self._parse_table(table))
        except Exception as e:
            print(f"Table extraction failed: {e}")
        
        return modules
    
    def _parse_table(self, table: List[List[str]]) -> List[Dict]:
        """
        Parse a table structure to extract module information.
        
        Args:
            table: Table data as list of lists
            
        Returns:
            List[Dict]: List of module dictionaries in the expected format
        """
        modules = []
        
        if not table or len(table) < 2:
            return modules
        
        # Try to identify column headers
        headers = [str(cell).lower() if cell else "" for cell in table[0]]
        
        # Common header patterns
        name_cols = [i for i, h in enumerate(headers) if any(keyword in h for keyword in ['name', 'titel', 'fach', 'modul'])]
        grade_cols = [i for i, h in enumerate(headers) if any(keyword in h for keyword in ['note', 'grade', 'bewertung'])]
        ects_cols = [i for i, h in enumerate(headers) if any(keyword in h for keyword in ['ects', 'cp', 'credits', 'leistungspunkte'])]
        
        # Process data rows
        for row_idx, row in enumerate(table[1:]):
            if not row or all(not cell for cell in row):
                continue
            
            module_info = {}
            
            # Extract module name
            if name_cols:
                module_info['Module'] = str(row[name_cols[0]]) if len(row) > name_cols[0] and row[name_cols[0]] else ""
            else:
                # Fallback: use first non-numeric column
                for cell in row:
                    if cell and not re.match(r'^\d+[,.]?\d*$', str(cell)):
                        module_info['Module'] = str(cell)
                        break
            
            # Extract grade
            if grade_cols:
                grade_text = str(row[grade_cols[0]]) if len(row) > grade_cols[0] and row[grade_cols[0]] else ""
                try:
                    module_info['Grade'] = float(grade_text.replace(',', '.'))
                except (ValueError, AttributeError):
                    module_info['Grade'] = None
            
            # Extract ECTS
            if ects_cols:
                ects_text = str(row[ects_cols[0]]) if len(row) > ects_cols[0] and row[ects_cols[0]] else ""
                try:
                    module_info['Credits'] = float(ects_text.replace(',', '.'))
                except (ValueError, AttributeError):
                    module_info['Credits'] = None
            
            # Estimate semester based on table position
            module_info['Semester'] = max(1, (row_idx // 10) + 1)  # Rough estimation
            
            # Only add if we have meaningful data
            if (module_info.get('Module') and 
                (module_info.get('Grade') is not None or module_info.get('Credits') is not None)):
                module_info['status'] = "bestanden" if module_info.get('Grade', 5) <= 4.0 else "nicht bestanden"
                
                # Create courses structure
                module_info['Courses'] = [{
                    'Course': module_info['Module'],
                    'Grade': module_info['Grade'],
                    'Credits': module_info['Credits'],
                    'Semester': module_info['Semester']
                }]
                
                modules.append(module_info)
        
        return modules
    
    def get_modules_dict(self) -> Dict[str, Dict]:
        """
        Get modules as a dictionary with module names as keys.
        
        Returns:
            Dict[str, Dict]: Dictionary with module names as keys and module info as values
        """
        if not self.modules:
            self.parse_modules()
        
        return {module['Module']: module for module in self.modules}
    
    def get_modules_dataframe(self) -> pd.DataFrame:
        """
        Get modules as a pandas DataFrame compatible with the analysis functions.
        
        Returns:
            pd.DataFrame: DataFrame with module information in the expected format
        """
        if not self.modules:
            self.parse_modules()
        
        # Import here to avoid circular import
        from .analysis import prepare_module_dataframe
        
        return prepare_module_dataframe(self.modules)
    
    def calculate_gpa(self, modules: Optional[List[Dict]] = None) -> float:
        """
        Calculate the GPA (Grade Point Average) based on ECTS-weighted grades.
        
        Args:
            modules: Optional list of modules to calculate GPA for. If None, uses all modules.
            
        Returns:
            float: Calculated GPA
        """
        if modules is None:
            modules = self.modules if self.modules else self.parse_modules()
        
        # Import here to avoid circular import
        from .analysis import calculate_gpa
        
        # Convert to DataFrame format expected by analysis function
        df = pd.DataFrame(modules)
        return calculate_gpa(df)
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of the transcript.
        
        Returns:
            Dict: Summary statistics including total ECTS, GPA, number of modules, etc.
        """
        if not self.modules:
            self.parse_modules()

        passed_modules = [m for m in self.modules
                          if m.get('Grade') is not None and m['Grade'] <= 4.0]

        # Count total courses
        total_courses = 0
        for module in self.modules:
            if 'Courses' in module and isinstance(module['Courses'], list):
                total_courses += len(module['Courses'])
            else:
                total_courses += 1

        return {
            'total_modules': len(self.modules),
            'total_courses': total_courses,
            'passed_modules': len(passed_modules),
            'failed_modules': len([m for m in self.modules
                                   if m.get('Grade') is not None and m['Grade'] > 4.0]),
            'total_ects': sum((m.get('Credits') or 0) for m in passed_modules),
            'gpa': self.calculate_gpa(),
            'best_grade': min((m['Grade'] for m in passed_modules), default=None),
            'worst_grade': max((m['Grade'] for m in passed_modules), default=None)
        }
    
    def export_to_excel(self, output_path: Union[str, Path]) -> None:
        """
        Export module data to Excel file.
        
        Args:
            output_path: Path for the output Excel file
        """
        df = self.get_modules_dataframe()
        df.to_excel(output_path, index=False)
    
    def __str__(self) -> str:
        """String representation of the transcript importer."""
        if not self.modules:
            return f"RWTHTranscriptImporter({self.pdf_path}) - Not parsed yet"
        
        stats = self.get_summary_stats()
        return (f"RWTHTranscriptImporter({self.pdf_path}) - "
                f"{stats['total_modules']} modules, "
                f"{stats['total_ects']} ECTS, "
                f"GPA: {stats['gpa']:.2f}")


def load_transcript(pdf_path: Union[str, Path]) -> RWTHTranscriptImporter:
    """
    Convenience function to load and parse a transcript PDF.
    
    Args:
        pdf_path: Path to the PDF transcript file
        
    Returns:
        RWTHTranscriptImporter: Initialized and parsed importer instance
    """
    importer = RWTHTranscriptImporter(pdf_path)
    importer.parse_modules()
    return importer