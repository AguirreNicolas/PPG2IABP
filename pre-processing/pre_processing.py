import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2 as pg2


def func_sql_filtered(parametros):

    ###### CONNECTION TO DB  ######
    h = parametros['host']
    db = parametros['database']
    u = parametros['user']
    psw = parametros['password'] 

    conn = pg2.connect(
    host = h,
    database = db,
    user = u,
    password = psw)
    #Cursor
    cursor = conn.cursor()
    

    ###### CHANNELS QUERY SQL  ######
    and_channels_selected = parametros['channels_selected_AND']
    or_channels_selected = parametros['channels_selected_OR']
    minutos_min = parametros['minutos_min'] #Cantidad minima de minutos
    MIN_LEN_MS = 1000 * 60 * minutos_min

    if len(or_channels_selected) is 0:
        text_sql = "SELECT * FROM files WHERE %s ::varchar[] <@ signals AND length_ms > %s AND type !='n';"
        query_sql = cursor.mogrify(text_sql,(and_channels_selected,MIN_LEN_MS,))
    else:
        COMBINE = []
        for o in or_channels_selected:
            elems = and_channels_selected + [o]
            elems = ["'" + e + "'" for e in elems]
            COMBINE.append(",".join(elems))
        query_sql = "SELECT * FROM files WHERE ({}) AND length_ms >= {} AND type !='n'".format(' OR '.join(["ARRAY[{}]::varchar[] <@ signals".format(e) for e in COMBINE]), MIN_LEN_MS)
    
    df = pd.read_sql_query(query_sql, conn)

    ###### AGREGO COLUMNA INDICE DE CANALES ######
    print(f'Agregando Columna Indice de Canales Solicitados ...')
    if len(or_channels_selected) is 0: # Only AND
        q_channels_selected = np.array(and_channels_selected).shape[0] # por el OR
        idx_channels_selected = np.zeros((q_channels_selected),dtype=list)
        acumm = np.zeros((q_channels_selected),dtype=list)
        for index, row in df.iterrows():
            #AND Channels
            for i_c in np.arange(0,q_channels_selected):
                idx_channels_selected[i_c] =  np.where(
                    np.asarray(df['signals'][index]) == and_channels_selected[i_c])[0]
        
            acumm = np.vstack((acumm,idx_channels_selected))
        acumm = acumm[1:]
        df['idx_signal'] = list(np.empty((df.shape[0],q_channels_selected)))
        df['idx_signal'] = list(acumm)

    else: # AND + OR
        q_channels_selected = np.array(and_channels_selected).shape[0] +1 # por el OR
        idx_channels_selected = np.zeros((q_channels_selected),dtype=list)
        acumm = np.zeros((q_channels_selected),dtype=list)
        for index, row in df.iterrows():
            #AND Channels
            for i_c in np.arange(0,q_channels_selected-1):
                idx_channels_selected[i_c] =  np.where(
                    np.asarray(df['signals'][index]) == and_channels_selected[i_c])[0]
            # OR Channels
            i_or = 0
            or_flag = 0;
            while or_flag == 0 and i_or <len(or_channels_selected):
                indx_or = np.where(
                    np.asarray(df['signals'][index]) == or_channels_selected[i_or])[0]
                if len(indx_or) != 0:
                    idx_channels_selected[-1] = indx_or # Last position OR
                    or_flag = 1
                else:
                    i_or = i_or + 1

            acumm = np.vstack((acumm,idx_channels_selected))
        acumm = acumm[1:]
        df['idx_signal'] = list(np.empty((df.shape[0],q_channels_selected)))
        df['idx_signal'] = list(acumm)

    print(f'Final Shape: {np.shape(df)}')
    
    ###### PLOT FINAL  ######
    plot_path = parametros['plot_path']
    select_col='signals'
    select_col = df.columns.isin([select_col])
    select_col = [i for i, x in enumerate(select_col) if x]
    df_pivot = df.iloc[:,select_col].astype(str)
    top_signals = pd.value_counts(df_pivot.values.flatten()).index[0:10]
    i_s = str(str(and_channels_selected) + ' OR ' + str(or_channels_selected))
    plt.figure(figsize=(15,3))
    g = sns.countplot(x='signals',data=df_pivot, palette = None, order=top_signals)
    g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
    plt.title(f'TOP 10 para {i_s}')
    plot_name = str(plot_path+'signal_count'+".eps")
    plt.savefig(plot_name, dpi=150,bbox_inches = "tight")
    plt.show()
    plt.figure(figsize=(15,3))
    ax = sns.distplot(df['length_ms']/(1000*60),  kde=False,hist=True)
    plt.xlabel('Min')
    plt.title(f'Duration dist [Min] {i_s}')
    plot_name = str(plot_path+'signal_duration'+".eps")
    plt.savefig(plot_name, dpi=150,bbox_inches = "tight")
    plt.show()

    return(df)