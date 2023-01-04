import geopandas as gpd
import pandas as pd

file= '/home/beland/PycharmProjects/gdb/Placettes/PET5_GPKG/PET5.gpkg'

placette = gpd.read_file(file, layer='placette')
dendro_gaules = gpd.read_file(file, layer='dendro_gaules')
dendro_arbres = gpd.read_file(file, layer='dendro_arbres')
pee_ori_sond = gpd.read_file(file, layer='pee_ori_sond')
placette_mes = gpd.read_file(file, layer='placette_mes')

allometric = pd.read_csv("/home/beland/PycharmProjects/Data_biomass/allometrique_biomass_indice_.csv")
#print(placette.head())
print(dendro_gaules.head())
print(dendro_arbres.head())
print(pee_ori_sond.head())
print(placette_mes.head())

dendro_arbres = dendro_arbres[dendro_arbres['dhp'].notna()]

placette_arbre = pd.merge(placette_mes[["id_pe", "date_sond", "id_pe_mes"]], dendro_arbres, on="id_pe_mes", how='left')
placette_gaules = pd.merge(placette_mes[["id_pe", "date_sond", "id_pe_mes"]], dendro_gaules, on="id_pe_mes", how='left')
placette_gaules = placette_gaules[placette_gaules['cl_dhp'].notna()]

placette_arbre['dhp_cm']= placette_arbre['dhp'].astype(float)*0.1
placette_gaules['dhp_cm']= placette_gaules['cl_dhp'].astype(int)

placette_alltiges =pd.concat([placette_arbre,placette_gaules], axis=0, ignore_index=True)
placette_alltiges['dhp_paire']=round(placette_alltiges['dhp_cm']/2)*2

tige_biomass_temp = pd.merge(placette_alltiges, allometric, left_on="essence", right_on='code_ess', how='left')

tige_biomass_temp['bio_compo_ha']= (tige_biomass_temp['a']*(tige_biomass_temp['dhp_cm']**tige_biomass_temp['b']))*tige_biomass_temp['tige_ha']

placette_biomass= tige_biomass_temp.groupby(by=["id_pe_mes", "date_sond", "id_pe_x"], as_index=False).sum()
placette_biomass= placette_biomass[["id_pe_mes", "date_sond", "id_pe_x", "bio_compo_ha"]]
placette_biomass['bio_compo_Mg']= placette_biomass["bio_compo_ha"]/1000

placette_biomass_info = pd.merge(placette_biomass, pee_ori_sond[["id_pe_mes","perturb", "an_perturb", "type_couv", "cl_dens","cl_haut", "cl_age"]], on="id_pe_mes", how='left')

placette_biomass_info_geom = pd.merge(placette[["id_pe","feuillet", "latitude", "longitude", "geometry"]], placette_biomass_info, left_on="id_pe", right_on="id_pe_x", how='left')

#test=placette_biomass_info[placette_biomass_info['id_pe_mes'] == '920340840203']

placette_biomass_info_geom.to_file("/home/beland/PycharmProjects/Data_biomass/Biomass_Placette/PET5.shp", driver='ESRI Shapefile')
