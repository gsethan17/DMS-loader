import os

data_dir = os.path.join(os.getcwd(), 'dms_rev0')
meta_path = os.path.join(os.getcwd(), 'dms_rev0', 'meta.csv')

total_drivers = ['GeesungOh', 'EuiseokJeong', 'TaesanKim']

driver_states = {
    0:'Unknown',
    1:'Angry/Disgusting',  #FF816E
    2:'Excited/Surprised',  #FFBC8D
    3:'Sad/Fatigued',  #C0E8FF
    4:'Happy/Neutral', #FFEC95
}

units={
    'ACC':"g",
    'BVP':"-",
    'EDA':"$\mu$S",
    'HR':"bpm", 
    'IBI':"-",
    'TEMP':"$^\circ$C",
    'CYL_PRES':"-",
    'LAT_ACCEL':"-",
    'LONG_ACCEL':"-",
    'YAW_RATE':"-",
    'WHL_SPD_FL':"kph",
    'WHL_SPD_FR':"kph",
    'WHL_SPD_RL':"kph",
    'WHL_SPD_RR':"kph",
    'CR_Brk_StkDep_Pc':"-",
    'CR_Ems_EngSpd_rpm':"-",
    'CR_Ems_FueCon_uL':"-",
    'CR_Ems_VehSpd_Kmh':"kph",
    'SAS_Angle':"$^\circ$",
    'CR_Hcu_HigFueEff_Pc':"-",
    'CR_Hcu_NorFueEff_Pc':"-",
    'CR_Fatc_OutTempSns_C':"-",
    'CR_Hcu_FuelEco_MPG':"-",
    'CR_Hcu_HevMod':"-",
    'CF_Ems_BrkForAct':"-",
    'CR_Ems_EngColTemp_C':"-",
    'CF_Clu_Odometer':"-",
    'BAT_SOC':"-",
}

cols_BIO = [
    'ACC',
    'BVP',
    'EDA',
    'HR',
    'IBI',
    'TEMP',
]

cols_CAN_conti = [
    'CYL_PRES',
    'LAT_ACCEL',
    'LONG_ACCEL',
    'YAW_RATE',
    'WHL_SPD_FL',
    'WHL_SPD_FR',
    'WHL_SPD_RL',
    'WHL_SPD_RR',
    'CR_Brk_StkDep_Pc',
    'CR_Ems_EngSpd_rpm',
    'CR_Ems_FueCon_uL',
    'CR_Ems_VehSpd_Kmh',
    'SAS_Angle',
    'CR_Hcu_HigFueEff_Pc',
    'CR_Hcu_NorFueEff_Pc',
    'CR_Fatc_OutTempSns_C',
    'CR_Hcu_FuelEco_MPG',
    'CR_Hcu_HevMod',
    'CF_Ems_BrkForAct',
    'CR_Ems_EngColTemp_C',
    'CF_Clu_Odometer',
    'BAT_SOC',
]
cols_CAN_cate = [
    'CR_Ems_AccPedDep_Pc',
    'CF_Ems_EngStat',
    'CF_Clu_InhibitD',
    'CF_Clu_InhibitN',
    'CF_Clu_InhibitP',
    'CF_Clu_InhibitR',
    'CF_Tcu_TarGe',
    'CF_Gway_HeadLampHigh',
    'CF_Gway_HeadLampLow',
    'CF_Hcu_DriveMode',
    'CR_Hcu_EcoLvl',
    'CF_Clu_VehicleSpeed',
]