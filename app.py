import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from pathlib import Path
import sys

from src.importer import RWTHTranscriptImporter, load_transcript, load_sample_data
from src.analysis import calculate_gpa, analyze_exclusions


# Page configuration
st.set_page_config(
    page_title="RWTH Notenstreicher",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def main():  
    # Title and header
    st.title("🎓 RWTH Notenstreicher")
    
    st.markdown("""
    Use this application to find out which module you could withdraw (“streichen”) to achieve the greatest possible improvement to your average grade if finishing within the regular study period (*Regelstudienzeit*).
                
    The official rules are described by the [RWTH Examinations Office](https://www.rwth-aachen.de/cms/root/studium/im-studium/pruefungsangelegenheiten/pruefungen/~whcqh/modulnotenstreichung-bei-regelstudienzei/?lidx=1).

    """)
    
    # Sidebar for data input
    st.sidebar.header("📊 Data Input")
    
    # Data input method selection
    input_method = st.sidebar.radio(
        "Choose data input method:",
        ["📁 Upload PDF Transcript", "✏️ Manual Entry", "📋 Use Sample Data"],
        format_func=lambda x: x  # Keep the display format with emojis
    )
    
    # Clean method names for code logic
    method_map = {
        "📁 Upload PDF Transcript": "upload_pdf",
        "✏️ Manual Entry": "manual_entry", 
        "📋 Use Sample Data": "sample_data"
    }
    method = method_map[input_method]
    
    df = None
    
    if method == "upload_pdf":
        st.sidebar.markdown("### PDF Upload")
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload your RWTH transcript PDF",
            type=['pdf'],
            help="Upload your official RWTH transcript PDF file"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with open("temp_transcript.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process PDF
                with st.spinner("Processing PDF transcript..."):
                    importer = RWTHTranscriptImporter("temp_transcript.pdf")
                    modules = importer.parse_modules()
                    df_from_pdf = importer.get_modules_dataframe()
                
                if len(modules) > 0:
                    st.sidebar.success(f"✅ Successfully imported {len(modules)} modules!")
                    
                    # Rename columns to match expected format
                    df = df_from_pdf.copy()
                    df = df.rename(columns={
                        'module_name': 'Module',
                        'grade': 'Grade', 
                        'ects': 'Credits'
                    })
                    
                    # Add semester info if missing
                    if 'Semester' not in df.columns:
                        df['Semester'] = range(1, len(df) + 1)
                    
                    # Clean up temp file
                    Path("temp_transcript.pdf").unlink(missing_ok=True)
                    
                    # Show summary
                    stats = importer.get_summary_stats()
                    st.sidebar.markdown("**Summary:**")
                    for key, value in stats.items():
                        st.sidebar.write(f"• {str(key).title()}: {value}")
                else:
                    st.sidebar.warning("No modules found in PDF. Try manual entry.")
                    
            except Exception as e:
                st.sidebar.error(f"Error processing PDF: {str(e)}")
                st.sidebar.info("Please try manual entry instead.")
    
    elif method == "manual_entry":
        st.sidebar.markdown("### Manual Data Entry")
        
        # Initialize session state for manual data
        if 'manual_modules' not in st.session_state:
            st.session_state.manual_modules = []
        
        with st.sidebar.expander("Add New Module", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                module_name = st.text_input("Module Name")
                grade = st.number_input("Grade", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
            with col2:
                credits = st.number_input("ECTS Credits", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
                semester = st.number_input("Semester", min_value=1, max_value=20, value=1, help="This represents the earliest semester of the module's courses")
            
            if st.button("Add Module"):
                if module_name:
                    st.session_state.manual_modules.append({
                        'Module': module_name,
                        'Grade': grade,
                        'Credits': credits,
                        'Semester': semester,
                        'Courses': [{'Course': module_name, 'Grade': grade, 'Credits': credits, 'Semester': semester}]
                    })
                    st.success("Module added!")
                else:
                    st.error("Please enter a module name.")
        
        # Show current modules
        if st.session_state.manual_modules:
            st.sidebar.markdown(f"**Current Modules ({len(st.session_state.manual_modules)}):**")
            for i, module in enumerate(st.session_state.manual_modules):
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.write(f"{module['Module']} - Grade: {module['Grade']}, ECTS: {module['Credits']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{i}"):
                        st.session_state.manual_modules.pop(i)
                        st.rerun()
            
            if st.sidebar.button("Clear All"):
                st.session_state.manual_modules = []
                st.rerun()
            
            # Create DataFrame from manual data
            df = pd.DataFrame(st.session_state.manual_modules)
        
        else:
            st.sidebar.info("No modules added yet. Add some modules to start analysis. Since only modules can be excluded, it's sufficient to add the modules you want to analyze.")
    
    else:  # Use Sample Data
        st.sidebar.markdown("### Sample Data")
        st.sidebar.info("Using sample RWTH transcript data for demonstration.")
        
        df = load_sample_data()
    
    # Main analysis section
    if df is not None and len(df) > 0:
        st.markdown("---")
        
        # Current academic overview
        st.header("📊 Current Academic Overview")
        
        current_gpa = calculate_gpa(df)
        total_ects = df['Credits'].sum()
        total_modules = len(df)
        
        # Calculate total courses by counting individual courses within modules
        total_courses = 0
        if 'Courses' in df.columns and df['Courses'].notna().any():
            for _, module in df.iterrows():
                if isinstance(module.get('Courses'), list):
                    total_courses += len(module['Courses'])
                else:
                    # Fallback: count as 1 course if no course details
                    total_courses += 1
        else:
            # Fallback: assume each module has 1 course
            total_courses = len(df)
        
        # Show different metrics based on input method
        if method == "manual_entry":
            # For manual entry, only show module-level information
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current GPA", f"{current_gpa:.4f}")
            with col2:
                st.metric("Total Modules", total_modules)
            with col3:
                st.metric("Total ECTS", f"{total_ects:.1f}")
            with col4:
                best_grade = df['Grade'].min()
                worst_grade = df['Grade'].max()
                st.metric("Grade Range", f"{best_grade:.1f} - {worst_grade:.1f}")
        else:
            # For PDF upload and sample data, show course information too
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Current GPA", f"{current_gpa:.4f}")
            with col2:
                st.metric("Total Modules", total_modules)
            with col3:
                st.metric("Total Courses", total_courses)
            with col4:
                st.metric("Total ECTS", f"{total_ects:.1f}")
            with col5:
                best_grade = df['Grade'].min()
                worst_grade = df['Grade'].max()
                st.metric("Grade Range", f"{best_grade:.1f} - {worst_grade:.1f}")
        
        # Semester breakdown
        st.subheader("Semester Breakdown")
        
        # Create a detailed breakdown showing courses by semester
        if method == "manual_entry":
            # For manual entry, show simple module breakdown
            semester_stats = df.groupby('Semester').apply(
                lambda group: pd.Series({
                    'GPA': calculate_gpa(group),
                    'ECTS': group['Credits'].sum(),
                    'Modules': len(group)
                })
            ).round(3)
            semester_stats.columns = ['GPA', 'ECTS', 'Modules']
            st.dataframe(semester_stats, use_container_width=True)
        elif 'Courses' in df.columns and df['Courses'].notna().any():
            # Expand courses for semester analysis
            semester_data = []
            for _, module in df.iterrows():
                if isinstance(module.get('Courses'), list):
                    for course in module['Courses']:
                        if isinstance(course, dict) and 'Semester' in course:
                            semester_data.append({
                                'Semester': course['Semester'],
                                'Grade': course.get('Grade', module['Grade']),
                                'Credits': course.get('Credits', 0),
                                'Module': module['Module'],
                                'Course': course.get('Course', module['Module'])
                            })
                else:
                    # Fallback for modules without course details
                    semester_data.append({
                        'Semester': module.get('Semester', 1),
                        'Grade': module['Grade'],
                        'Credits': module['Credits'],
                        'Module': module['Module'],
                        'Course': module['Module']
                    })
            
            if semester_data:
                semester_df = pd.DataFrame(semester_data)
                semester_stats = semester_df.groupby('Semester').apply(
                    lambda group: pd.Series({
                        'GPA': calculate_gpa(group),
                        'ECTS': group['Credits'].sum(),
                        'Courses': len(group)
                    })
                ).round(3)
                semester_stats.columns = ['GPA', 'ECTS', 'Courses']
                st.dataframe(semester_stats, use_container_width=True)
            else:
                # Fallback to simple semester breakdown
                semester_stats = df.groupby('Semester').apply(
                    lambda group: pd.Series({
                        'GPA': calculate_gpa(group),
                        'ECTS': group['Credits'].sum(),
                        'Modules': len(group)
                    })
                ).round(3)
                semester_stats.columns = ['GPA', 'ECTS', 'Modules']
                st.dataframe(semester_stats, use_container_width=True)
        else:
            # Fallback for DataFrames without course information
            semester_stats = df.groupby('Semester').apply(
                lambda group: pd.Series({
                    'GPA': calculate_gpa(group),
                    'ECTS': group['Credits'].sum(),
                    'Modules': len(group)
                })
            ).round(3)
            semester_stats.columns = ['GPA', 'ECTS', 'Modules']
            st.dataframe(semester_stats, use_container_width=True)
        
        # Expander for detailed course list (only show for non-manual entry modes)
        if method != "manual_entry":
            with st.expander("Detailed Course List", expanded=False):
                # Create an expanded view showing courses and their modules
                if 'Courses' in df.columns and df['Courses'].notna().any():
                    expanded_data = []
                    for _, module in df.iterrows():
                        if isinstance(module.get('Courses'), list):
                            for course in module['Courses']:
                                if isinstance(course, dict):
                                    expanded_data.append({
                                        'Course': course.get('Course', 'N/A'),
                                        'Module': module['Module'],
                                        'Grade': course.get('Grade', module['Grade']),
                                        'Credits': course.get('Credits', 0),
                                        'Semester': course.get('Semester', 'N/A')
                                    })
                        else:
                            # Fallback for modules without course details
                            expanded_data.append({
                                'Course': module['Module'],
                                'Module': module['Module'],
                                'Grade': module['Grade'],
                                'Credits': module['Credits'],
                                'Semester': module.get('Semester', 'N/A')
                            })
                    
                    if expanded_data:
                        expanded_df = pd.DataFrame(expanded_data)
                        st.dataframe(expanded_df, use_container_width=True)
                    else:
                        st.dataframe(df, use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
        else:
            # For manual entry, show a simple module list
            with st.expander("📋 Module List", expanded=False):
                display_df = df[['Module', 'Grade', 'Credits', 'Semester']].copy()
                st.dataframe(display_df, use_container_width=True)

        st.markdown("---")

        # Analysis
        st.header("🔍 Module Exclusion Analysis")
        
        with st.spinner("Analyzing optimal exclusions..."):
            improvements_df, current_gpa = analyze_exclusions(df)
        
        # Top recommendation
        best_option = improvements_df.iloc[0]
        
        if best_option['Improvement'] > 0:
            st.success(f"""
            **OPTIMAL RECOMMENDATION**
            
            **Module to exclude:** {best_option['Module']}
            - **Module Grade:** {best_option['Grade']:.1f}
            - **Module ECTS Credits:** {best_option['Credits']:.1f}
            - **New GPA:** {best_option['New_GPA']:.4f}
            - **Improvement:** {best_option['Improvement']:.4f} points
            """)
        else:
            st.warning("No beneficial exclusions found with current grades.")
        
        # Top 3 recommendations table
        st.subheader("Top 3 Exclusion Options")
        top_3 = improvements_df.head(3)[['Rank', 'Module', 'Grade', 'Credits', 'New_GPA', 'Improvement']]
        
        st.dataframe(top_3, use_container_width=True, hide_index=True)
        
        st.markdown("---")

        # Next steps
        st.header("🚀 Next Steps")

        st.markdown(
            """
            1. **Review the recommendations** listed above.
            2. **Download and complete the official request form** from the [RWTH Examinations Office](https://www.rwth-aachen.de/cms/root/studium/im-studium/pruefungsangelegenheiten/pruefungen/~whcqh/modulnotenstreichung-bei-regelstudienzei/?lidx=1).
            3. **Submit the form** to your [faculty examinations office](https://www.rwth-aachen.de/go/id/cbcn) for approval.
            """
        )
    
    else:
        st.info("👈 Please select a data input method from the sidebar to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    Created by [Julian Winking](https://github.com/julianwinking/rwth-notenstreicher)
    """)

if __name__ == "__main__":
    main()
