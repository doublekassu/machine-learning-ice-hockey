import streamlit as st
from machine_learning_teams import *

#Valintasivu
st.title("🏒 NHL Tulosveikkaaja")
st.subheader("Valitse ennustettava kohde:")

# Valitaan kumpaa ennustetaan
valinta = st.selectbox("Mitä haluat ennustaa?", ["Lopputulosten tarkkuutta", "Pelaajan maalin todennäköisyys"])

# Ennustelomake ottelun voittajalle
if valinta == "Lopputulosten tarkkuutta":
    st.header("Muokkaa harjoitusdatan attribuutteja")
    "(kaudet 2008-2024)"
    
    harjoitus_aloituskausi = st.text_input("Harjoitusdatan aloituskausi")
    harjoitus_paatoskausi = st.text_input("Harjoitusdatan päätöskausi")

    st.header("Muokkaa testidatan attribuutteja")

    testi_aloituskausi = st.text_input("Testidatan aloituskausi")
    testi_paatoskausi = st.text_input("Testidatan päätöskausi")

    # Ennustuspainike
    if st.button("Tarkista pelien ennustustarkkuus valituille kausille"):
        try:
            # Muutetaan syötteet kokonaisluvuiksi
            h_alku = int(harjoitus_aloituskausi)
            h_loppu = int(harjoitus_paatoskausi)
            t_alku = int(testi_aloituskausi)
            t_loppu = int(testi_paatoskausi)

            # Tarkistetaan syötteiden loogisuus
            if h_alku < h_loppu and t_alku >= h_loppu and t_alku < t_loppu:
                default_attributes, extra_attributes = machine_learning(h_alku, h_loppu, t_alku, t_loppu)
                st.success(f"Algoritmi ennusti {default_attributes}% ajasta lopputuloksen oikein tarkasteltavan joukkueen näkökulmasta.\n"
                    "Attribuutteina olivat playoffit, koti- vai vieraspeli sekä pelaavat joukkueet.")
                
                st.success(f"Algoritmi ennusti {extra_attributes}% ajasta oikein pelin voittajan ja häviäjän.\n"
                    "Attribuutteina käytettiin yllämainittujen lisäksi joukkekohtaisesti keskiarvoja viimeisen kolmen pelin ajalta. "
                    "Lisäattribuutit olivat tehdyt ja päästetyt maalit, vedot ja jäähyminuutit molemmin puolin.")
            else:
                st.error("Syöttämäsi kausirajat eivät ole loogiset. Tarkista, että:\n- Harjoitusdatan alku < loppu\n- Testidata alkaa harjoitusdatan jälkeen\n- Testidatan alku < loppu")

        except ValueError:
            st.error("Anna kaikki kausiarvot kokonaislukuina väliltä 2008-2025")

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