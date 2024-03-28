import streamlit as st
from pycaret.regression import load_model
pipeline = load_model("best_model")
