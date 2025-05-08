import streamlit as st
from machine_learning_teams import *
from goal_predictor import *

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
    st.header("Syötä pelaajan ID")
    pelaaja_id = st.text_input("Pelaajan ID")
    
    # Ennustuspainike
    if st.button("Laske maalin todennäköisyys"):
        if pelaaja_id:
            try:
                # Convert input to integer
                pelaaja_id = int(pelaaja_id)
                
                # Load data and train model
                df_train, df_2023 = load_data()
                model, features, target = train_model(df_train)
                
                # First try to find player in 2023 data (for actual results)
                player_row_2023 = df_2023[df_2023["playerId"] == pelaaja_id]
                
                if player_row_2023.empty:
                    st.warning(f"Pelaajaa ID:llä '{pelaaja_id}' ei löytynyt kaudelta 2023.")
                else:
                    player_name = player_row_2023["name"].values[0]
                    actual_goals = player_row_2023["I_F_goals"].values[0]
                    
                    # Find player in training data to make prediction
                    player_data_from_train = df_train[df_train["playerId"] == pelaaja_id]
                    
                    if player_data_from_train.empty:
                        st.warning(f"Pelaajalla {player_name} (ID: {pelaaja_id}) ei ole dataa harjoituskausilta 2019-2022.")
                    else:
                        # Use most recent season data from training set
                        player_features = player_data_from_train.sort_values('season', ascending=False).iloc[0][features]
                        
                        # Make prediction
                        predicted_goals = model.predict(player_features.values.reshape(1, -1))[0]
                        
                        # Calculate accuracy
                        accuracy = 100 - abs(predicted_goals - actual_goals) / actual_goals * 100 if actual_goals > 0 else 0
                        
                        # Display results
                        st.write(f"Pelaaja: {player_name} (ID: {pelaaja_id})")
                        st.success(f"Ennustetut maalit (2023): {predicted_goals:.2f}")
                        st.write(f"Toteutuneet maalit (2023): {actual_goals:.0f}")
                        st.write(f"Ennusteen tarkkuus: {accuracy:.2f}%")
                        
            except ValueError:
                st.error("Anna pelaajan ID kokonaislukuna")
        else:
            st.warning("Syötä pelaajan ID ensin")
                    