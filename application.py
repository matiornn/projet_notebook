"""
📝 **Instructions** :
- Installez toutes les bibliothèques nécessaires en fonction des imports présents dans le code, utilisez la commande suivante :conda create -n projet python pandas numpy ..........
- Complétez les sections en écrivant votre code où c’est indiqué.
- Ajoutez des commentaires clairs pour expliquer vos choix.
- Utilisez des emoji avec windows + ;
- Interprétez les résultats de vos visualisations (quelques phrases).
"""

### 1. Importation des librairies et chargement des données
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import kagglehub

# Chargement des données
#df = pd.read_csv("........ds_salaries.csv")

path = kagglehub.dataset_download("arnabchaki/data-science-salaries-2023")
print("Path to dataset files:", path)

# Lire le fichier depuis le répertoire du projet

df = pd.read_csv(f"{path}\ds_salaries.csv")
print(df.head(10))  # Affichage des 10 premières lignes du DataFrame


### 2. Exploration visuelle des données
#votre code 

st.title("📊 Visualisation des Salaires en Data Science")
st.markdown("Explorez les tendances des salaires à travers différentes visualisations interactives.")

if st.checkbox("Afficher un aperçu des données"):
    st.write(df.head(10))

#Statistique générales avec describe pandas 
#votre code 

st.subheader("📌 Statistiques générales")
st.write(df.describe())

### 3. Distribution des salaires en France par rôle et niveau d'expérience, uilisant px.box et st.plotly_chart
#votre code 

st.title("📈 Distribution des salaires en France")

df_France = df[df.company_location == 'FR'] # filtrage sur la France

st.subheader('Distribution des salaires en France par rôle')
selected_option = st.selectbox("Choisir un métier", options=df_France['job_title'].unique()) # Création du bouton permettant le filtrage sur les métiers 
distri_salaire_role = px.box(df_France[df_France['job_title'] == selected_option], x='job_title', y='salary_in_usd')  # Boxplot interactif
st.plotly_chart(distri_salaire_role.update_layout(xaxis_title="Métiers", yaxis_title="Salaires en USD")) # Changement des noms des axes

# Interprétation
st.markdown("Interprétation : ")
st.markdown("Ce graphique nous permet de découvrir un boxplot par métiers et permet de mieux situer la fourchette salariale de celui-ci. Autrement dit, cela permet de constater la distribution des salaires au sein d'un même métiers.")

st.subheader("Distribution des salaires en France par niveau d'expérience") 
distri_salaire_exp = px.box(df_France, x='experience_level', y='salary_in_usd', color = 'experience_level')
st.plotly_chart(distri_salaire_exp.update_layout(xaxis_title="Niveaux d'expérience", yaxis_title="Salaires en USD"))  # Changement des noms des axes

# Interprétation
st.markdown("Interprétation : ")
st.markdown("Ce graphique nous permet de découvrir un boxplot par niveau d'expérience et permet de mieux situer la fourchette salariale de ceux-ci. Autrement dit, cela permet de constater la distribution des salaires au sein des différents niveaux d'expériences. Il permet aussi de mettre en lumière des invidus hors normes ou peut-être abérrants.")

### 4. Analyse des tendances de salaires :
#### Salaire moyen par catégorie : en choisisant une des : ['experience_level', 'employment_type', 'job_title', 'company_location'], utilisant px.bar et st.selectbox 

st.subheader('Salaire moyen par catégorie')
selected_option_bis = st.selectbox("Choisir une option", options=['experience_level', 'employment_type', 'job_title', 'company_location']) # Création du bouton permettant le filtrage sur l'axe à représenter
tendance_salaire = px.bar(df_France.groupby(selected_option_bis)['salary_in_usd'].mean().reset_index(), x=selected_option_bis, y='salary_in_usd') 
st.plotly_chart(tendance_salaire.update_layout(yaxis_title="Salaires en USD"))

# Interprétation
st.markdown("Interprétation : ")
st.markdown("Ce graphique et ce filtre nous permet de constater les écarts de salaire moyen selon son niveau d'expérience, ou sa qualification, ou son métier ou encore le lieu de l'entreprise.")

### 5. Corrélation entre variables
# Sélectionner uniquement les colonnes numériques pour la corrélation
#votre code 

numeric_df = df.select_dtypes(include=[np.number])  # Sélection des colonnes numériques uniquement

# Calcul de la matrice de corrélation
#votre code

matrice_corr = numeric_df.corr()

# Affichage du heatmap avec sns.heatmap
#votre code 

heatmap, ax = plt.subplots(figsize=(10, 8)) # 

sns.heatmap(matrice_corr, annot=True, cmap='coolwarm', ax=ax) # Création de la heatmap

st.subheader("🔗 Corrélations entre variables numériques")
st.pyplot(heatmap)

# Interprétation
st.markdown("Interprétation : ")
st.markdown("Cette heatmap représente la corrélation entre les différentes variables numériques de notre jeu de données. Elle montre que les deux variables quantitatives les plus correlées sont le salaire et le télé-travail.")

### 6. Analyse interactive des variations de salaire
# Une évolution des salaires pour les 10 postes les plus courants
# count of job titles pour selectionner les postes
# calcule du salaire moyen par an
#utilisez px.line
#votre code 

st.subheader("Analyse interactive des variations de salaire")

# 10  poste les plus fréquents (filtrage)

top10_poste = df.groupby('job_title')['job_title'].count().sort_values(ascending=False).head(10).index
df_top10 = df[df['job_title'].isin(top10_poste)]

# Création du graphique 
salair_moy_job_year = df_top10.groupby(['work_year', 'job_title'])['salary_in_usd'].mean().reset_index()

salaire_moy_an = px.line(salair_moy_job_year, x='work_year', y='salary_in_usd', color='job_title')

st.plotly_chart(salaire_moy_an.update_layout(xaxis_title="Année", yaxis_title="Salaires moyen en USD"))

# Interprétation
st.markdown("Interprétation : ")
st.markdown("Ce graphique met en évidence l'évolution du salaire moyen des 10 métiers de la data les plus représentés. Globalement, tous les métiers suivent la même évolution hormis les data architect et apply scientist qui restent quasiment constant. Les research engineer connaissent un déclin.")

### 7. Salaire médian par expérience et taille d'entreprise
# utilisez median(), px.bar
#votre code 

st.subheader("Salaire médian par expérience et taille d'entreprise")
salair_med_exp_taille = df.groupby(['experience_level', 'company_size'])['salary_in_usd'].median().reset_index()
graph_salair_med_exp_taille = px.bar(salair_med_exp_taille, x='experience_level', y='salary_in_usd', color = 'company_size')
st.plotly_chart(graph_salair_med_exp_taille.update_layout(xaxis_title="Niveau d'expérience", yaxis_title="Salaires médian en USD"))

# Interprétation
st.markdown("Interprétation : ")
st.markdown("Ce graphique montre les salaires médians en fonction du niveau d'expérience et de la taille de l'entreprise. On remarque que le plus haut salaire médian sont les travailleurs expérimenté et qui travaille dans une entreprise de taille moyenne. Plus généralement, ce sont les entreprises de moyennes et grandes tailles qui payent le mieux.")

### 8. Ajout de filtres dynamiques
#Filtrer les données par salaire utilisant st.slider pour selectionner les plages 
#votre code 

plage_salaire = st.slider("Choisir une plage", value = (df['salary_in_usd'].min() , df['salary_in_usd'].max()))


### 9.  Impact du télétravail sur le salaire selon le pays

st.subheader("Impact du télétravail sur le salaire selon le pays")
teletravail_salaire_pays = df.groupby(['remote_ratio', 'company_location'])['salary_in_usd'].median().reset_index()
graph_teletravail_salaire_pays = px.bar(teletravail_salaire_pays, x='company_location', y='salary_in_usd', color = 'remote_ratio')
st.plotly_chart(graph_teletravail_salaire_pays.update_layout(xaxis_title="Pays", yaxis_title="Salaires médian en USD"))

# Interprétation
st.markdown("Interprétation : ")
st.markdown("Ce graphique nous montre l'impact du télé-travail sur le salaire en fonction du pays. On remarque que dans nos données certains pays ne font que du télé-travail ce qui semble peu problable et remet en cause la réelle fiabilité des données.")

### 10. Filtrage avancé des données avec deux st.multiselect, un qui indique "Sélectionnez le niveau d'expérience" et l'autre "Sélectionnez la taille d'entreprise"
#votre code 

nv_exp = st.multiselect("Choisir un niveau d'expérience", df['experience_level'].unique(), default=df['experience_level'].unique())

taille_entreprise = st.multiselect("Choisir une taille d'entreprise", df['company_size'].unique(), default=df['company_size'].unique())