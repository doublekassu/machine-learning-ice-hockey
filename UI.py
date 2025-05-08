import streamlit as st
from machine_learning_teams import *

#Valintasivu
st.title("🏒 NHL Tulosveikkaaja")
st.subheader("Valitse ennustettava kohde:")

# Valitaan kumpaa ennustetaan
valinta = st.selectbox("Mitä haluat ennustaa?", ["Ottelun voittaja", "Pelaajan maalin todennäköisyys"])

# Ennustelomake ottelun voittajalle
if valinta == "Ottelun voittaja":
    st.header("Muokkaa harjoitusdatan attribuutteja")
    
    harjoitus_aloituskausi = st.text_input("Harjoitusdatan aloituskausi")
    harjoitus_paatoskausi = st.text_input("Harjoitusdatan päätöskausi")
    
    st.header("Muokkaa testidatan attribuutteja")

    testi_aloituskausi = st.text_input("Testidatan aloituskausi")
    testi_paatoskausi = st.text_input("Testidatan päätöskausi")

    # Ennustuspainike
    if st.button("Veikkaa voittaja"):
        #test.trainData(harjoitus_aloituskausi, harjoitus_paatoskausi)
        #test.testData(testi_aloituskausi, testi_paatoskausi)

        #st.success("Kausina ", testi_aloituskausi " - ", testi_paatoskausi, " algoritmi ennusti ottelun voittajan ja häviäjän oikein ", accuracy, " tarkkuudella")
        st.success("pöö")

# Ennustelomake pelaajan maalille
elif valinta == "Pelaajan maalin todennäköisyys":
    st.header("🥅 Ennusta pelaajan maalin todennäköisyys")
    
    # Syöte: pelkkä pelaajan nimi
    pelaaja = st.text_input("Pelaajan nimi")

    # Ennustuspainike
    if st.button("Laske maalin todennäköisyys"):
        st.write("Syötetyt tiedot:")
        st.write(f"Pelaaja: {pelaaja}")
        st.success("(Tähän tulee maalin todennäköisyys myöhemmin)")