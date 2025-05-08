import streamlit as st
from machine_learning_teams import *
from goal_predictor import *

#Valintasivu
st.title("🏒 NHL Tulosveikkaaja")
st.subheader("Valitse ennustettava kohde:")

# Valitaan kumpaa ennustetaan
valinta = st.selectbox("Mitä haluat ennustaa?", ["Lopputulosten tarkkuutta", "Maalien määrää"])

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
elif valinta == "Maalien määrää":
    st.header("Syötä pelaajan ID")
    pelaaja_id = st.text_input("Pelaajan ID")
    
    # Ennustuspainike
    if st.button("Laske maalien todennäköisyys"):
        if pelaaja_id:
            try:
                # Convert input to integer
                pelaaja_id = int(pelaaja_id)
                
                # Load data and train model
                df_train, df_2023 = load_data()
                model, features, target = train_model(df_train)
                
                # Run prediction function
                player_name, predicted_goals, actual_goals, accuracy, error_msg = predict_player_goals(
                    model, features, df_train, df_2023, pelaaja_id
                )
                
                r2, rmse, actual_mean, pred_mean = evaluate_model(model, features, df_2023)
                
                if error_msg:
                    st.warning(error_msg)
                else:
                    # Display results
                    st.write(f"Pelaaja: {player_name} (ID: {pelaaja_id})")
                    st.success(f"Ennustetut maalit kaudelle 2023 kausien 2019-2022 perusteella: {predicted_goals:.2f}")
                    st.success(f"Toteutuneet maalit kaudella 2023: {actual_goals:.0f}")
                    st.success(f"Ennusteen tarkkuus: {accuracy:.2f}%")
                    st.write("Kokonaistarkkuus kaikkien pelaajien kohdalla:", f"{r2:.4f}%")
                    st.write("Keskimääräinen ennustevirhe +-", f"{rmse:.4f}","maalia")

                    
            except ValueError:
                st.error("Anna pelaajan ID kokonaislukuna")
        else:
            st.warning("Syötä pelaajan ID ensin")
