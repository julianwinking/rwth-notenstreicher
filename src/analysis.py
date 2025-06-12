"""
Analysis functions for RWTH Notenstreicher.

This module contains functions for calculating GPA and analyzing module exclusions
to help students optimize their academic performance.
"""

import pandas as pd


def calculate_gpa(data):
    """Calculate weighted average grade based on ECTS credits."""
    if len(data) == 0:
        return 0.0
    return (data["Grade"] * data["Credits"]).sum() / data["Credits"].sum()


def analyze_exclusions(df):
    """Analyze all possible module exclusions and return sorted results."""
    current_gpa = calculate_gpa(df)
    improvements = []
    
    for i in range(len(df)):
        module = df.iloc[i]
        df_temp = df.drop(index=i)
        new_gpa = calculate_gpa(df_temp)
        improvement = current_gpa - new_gpa
        
        # Get semester info from courses if available
        semester = get_module_semester(module)
        
        improvements.append({
            'Rank': i + 1,
            'Module': module['Module'],
            'Grade': module['Grade'],
            'Credits': module['Credits'],
            'Semester': semester,
            'New_GPA': new_gpa,
            'Improvement': improvement,
            'Improvement_Percent': (improvement / current_gpa) * 100 if current_gpa > 0 else 0
        })
    
    improvements_df = pd.DataFrame(improvements)
    improvements_df = improvements_df.sort_values('Improvement', ascending=False)
    improvements_df['Rank'] = range(1, len(improvements_df) + 1)
    
    return improvements_df, current_gpa


def get_module_semester(module_data):
    """Get the earliest semester from the courses within a module."""
    if 'Courses' in module_data and isinstance(module_data['Courses'], list):
        semesters = [course.get('Semester') for course in module_data['Courses'] if course.get('Semester') is not None]
        if semesters:
            return min(semesters)
    
    # Fallback if no semester info in courses
    return module_data.get('Semester', 1)


def get_courses_for_module(module_data):
    """Extract course information from a module."""
    if 'Courses' in module_data and isinstance(module_data['Courses'], list):
        return module_data['Courses']
    return []


def calculate_module_grade_from_courses(courses):
    """Calculate module grade from individual course grades."""
    if not courses:
        return 0.0
    
    total_points = 0
    total_credits = 0
    
    for course in courses:
        if 'Grade' in course and 'Credits' in course:
            total_points += course['Grade'] * course['Credits']
            total_credits += course['Credits']
    
    return total_points / total_credits if total_credits > 0 else 0.0


def prepare_module_dataframe(modules_data):
    """Convert modules data with courses into a DataFrame with semester information."""
    processed_modules = []
    
    for module in modules_data:
        module_copy = module.copy()
        # Add semester info derived from courses
        module_copy['Semester'] = get_module_semester(module)
        processed_modules.append(module_copy)
    
    return pd.DataFrame(processed_modules)
