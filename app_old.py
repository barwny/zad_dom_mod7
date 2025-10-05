import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px

MODEL_NAME = "welcome_survey_clustering_pipeline_v2"

CLUSTER_NAMES_AND_DECRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

DATA = "welcome_survey_simple_v2.csv"

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open (CLUSTER_NAMES_AND_DECRIPTIONS, "r") as f:
        return json.loads(f.read())


@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=";")
    df_with_clusters = predict_model(model, data=all_df)
    
    return df_with_clusters

with st.sidebar:
    st.header("Znajdź osoby o podobnych zainteresowniach")
    st.markdown("Używając poniższych fitrów opisz siebie")
    gender = st.radio("Płeć", ['Kobieta', 'Mężczyzna'])
    age=st.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65',  'unknown'])
    edu_level=st.selectbox("Wyksztacenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Koty i Psy', 'Inne'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
if predicted_cluster_data['name'] == "Miłośnicy Innych Zwierząt":
    st.image("1.png", use_container_width=False)

if predicted_cluster_data['name'] == "Mężczyźni Nad Wodą":
    st.image("2.png", use_container_width=False)

if predicted_cluster_data['name'] == "Leśni Profesjonaliści":
    st.image("3.png", use_container_width=False)

if predicted_cluster_data['name'] == "Kociarze Górscy":
    st.image("4.png", use_container_width=False)

if predicted_cluster_data['name'] == "Górscy Miłośnicy Psów":
    st.image("5.png", use_container_width=False)

if predicted_cluster_data['name'] == "Górscy Profesjonaliści":
    st.image("6.png", use_container_width=False)

if predicted_cluster_data['name'] == "Młodzi Miłośnicy Gór":
    st.image("7.png", use_container_width=False)

if predicted_cluster_data['name'] == "Doświadczeni Miłośnicy Wody":
    st.image("nocd.png", use_container_width=False)

st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Zobacz rozkład w swojej grupy dla:")
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Płci", "Wieku", "Wykształcenia", "Ulubionych zwierząt",
                                         "Ulubionych miejsc"])

with tab0:
    fig = px.histogram(same_cluster_df, x="gender")
    fig.update_layout(
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab1:
    fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
    fig.update_layout(
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab2:
    fig = px.histogram(same_cluster_df, x="edu_level")
    fig.update_layout(
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab3:
    fig = px.histogram(same_cluster_df, x="fav_animals")
    fig.update_layout(
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab4:
    fig = px.histogram(same_cluster_df, x="fav_place")
    fig.update_layout(
        xaxis_title="Ulubione miejsce",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)



