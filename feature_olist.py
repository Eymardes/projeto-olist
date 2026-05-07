import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ── coloque o caminho da pasta onde estão os CSVs ───────────
PASTA = "C:/Users/eymar/Downloads/olist/archive/"

# ── carrega todas as tabelas ─────────────────────────────────
orders    = pd.read_csv(PASTA + 'olist_orders_dataset.csv')
items     = pd.read_csv(PASTA + 'olist_order_items_dataset.csv')
products  = pd.read_csv(PASTA + 'olist_products_dataset.csv')
customers = pd.read_csv(PASTA + 'olist_customers_dataset.csv')
sellers   = pd.read_csv(PASTA + 'olist_sellers_dataset.csv')
payments  = pd.read_csv(PASTA + 'olist_order_payments_dataset.csv')
reviews   = pd.read_csv(PASTA + 'olist_order_reviews_dataset.csv')
geo       = pd.read_csv(PASTA + 'olist_geolocation_dataset.csv')
traducao  = pd.read_csv(PASTA + 'product_category_name_translation.csv')

# ── converte colunas de data de string para datetime ─────────
for col in ['order_purchase_timestamp', 'order_approved_at',
            'order_delivered_carrier_date', 'order_delivered_customer_date',
            'order_estimated_delivery_date']:
    orders[col] = pd.to_datetime(orders[col])

items['shipping_limit_date'] = pd.to_datetime(items['shipping_limit_date'])

# ── mantém só pedidos entregues e sem datas faltando ─────────
orders = orders[orders['order_status'] == 'delivered'].copy()
orders = orders.dropna(subset=['order_delivered_customer_date',
                                'order_approved_at',
                                'order_delivered_carrier_date'])

# ── cria o target: quantos dias o pedido demorou ─────────────
orders['dias_entrega'] = (
    orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']
).dt.total_seconds() / 86400

# remove pedidos negativos ou absurdamente longos (outliers)
orders = orders[(orders['dias_entrega'] > 0) & (orders['dias_entrega'] <= 120)]

# ── features extraídas do timestamp da compra ────────────────
orders['hora_compra']              = orders['order_purchase_timestamp'].dt.hour
orders['dia_semana_compra']        = orders['order_purchase_timestamp'].dt.dayofweek
orders['mes_compra']               = orders['order_purchase_timestamp'].dt.month
orders['eh_fim_de_semana']         = (orders['dia_semana_compra'] >= 5).astype(int)

# tempo entre cada etapa do pedido em horas
orders['horas_ate_aprovacao']      = (
    orders['order_approved_at'] - orders['order_purchase_timestamp']
).dt.total_seconds() / 3600

orders['horas_ate_transportadora'] = (
    orders['order_delivered_carrier_date'] - orders['order_approved_at']
).dt.total_seconds() / 3600

# ── agrega itens por pedido (order_items tem 1 linha por item) ──
feats_itens = items.groupby('order_id').agg(
    qtd_itens          = ('order_item_id', 'count'),
    qtd_vendedores     = ('seller_id',     'nunique'),
    valor_total_pedido = ('price',         'sum'),
    valor_total_frete  = ('freight_value', 'sum'),
).reset_index()

feats_itens['multiplos_vendedores'] = (feats_itens['qtd_vendedores'] > 1).astype(int)
feats_itens['proporcao_frete']      = (
    feats_itens['valor_total_frete'] / feats_itens['valor_total_pedido']
).round(4)

# ── adiciona tradução e calcula volume, depois agrega por pedido ──
products = products.merge(traducao, on='product_category_name', how='left')
products['product_category_name_english'] = products['product_category_name_english'].fillna('desconhecido')
products['volume_cm3']    = products['product_length_cm'] * products['product_height_cm'] * products['product_width_cm']
products['tem_descricao'] = (products['product_description_lenght'] > 0).astype(int)

items_prod = items.merge(products, on='product_id', how='left')

feats_prod = items_prod.groupby('order_id').agg(
    peso_total_gramas = ('product_weight_g',              'sum'),
    volume_total_cm3  = ('volume_cm3',                    'sum'),
    media_qtd_fotos   = ('product_photos_qty',            'mean'),
    categoria_produto = ('product_category_name_english', lambda x: x.mode()[0] if len(x) > 0 else 'desconhecido'),
    tem_descricao     = ('tem_descricao',                 'max'),
).reset_index()

feats_prod['peso_por_item']     = (feats_prod['peso_total_gramas'] / feats_prod['peso_total_gramas'].replace(0, np.nan)).fillna(0)
feats_prod['peso_total_gramas'] = feats_prod['peso_total_gramas'].fillna(feats_prod['peso_total_gramas'].median())
feats_prod['volume_total_cm3']  = feats_prod['volume_total_cm3'].fillna(feats_prod['volume_total_cm3'].median())
feats_prod['media_qtd_fotos']   = feats_prod['media_qtd_fotos'].fillna(feats_prod['media_qtd_fotos'].median())

# ── calcula distância real em km entre vendedor e cliente ─────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# geolocation tem ~1M linhas (vários pontos por CEP) → média por CEP
geo_media = geo.groupby('geolocation_zip_code_prefix')[
    ['geolocation_lat', 'geolocation_lng']].mean().reset_index()

# coordenadas do cliente
cust_geo = customers.merge(
    geo_media, left_on='customer_zip_code_prefix',
    right_on='geolocation_zip_code_prefix', how='left'
).rename(columns={'geolocation_lat': 'lat_cliente', 'geolocation_lng': 'lng_cliente'})

# vendedor principal por pedido (primeiro item)
seller_por_pedido = items[['order_id', 'seller_id']].drop_duplicates('order_id')

# coordenadas do vendedor
seller_geo = seller_por_pedido.merge(sellers, on='seller_id', how='left').merge(
    geo_media, left_on='seller_zip_code_prefix',
    right_on='geolocation_zip_code_prefix', how='left'
).rename(columns={'geolocation_lat': 'lat_vendedor', 'geolocation_lng': 'lng_vendedor'})

# junta tudo e calcula distância + flags geográficas
geo_feats = orders[['order_id', 'customer_id']].merge(
    cust_geo[['customer_id', 'customer_state', 'lat_cliente', 'lng_cliente']],
    on='customer_id', how='left'
).merge(
    seller_geo[['order_id', 'seller_id', 'seller_state', 'lat_vendedor', 'lng_vendedor']],
    on='order_id', how='left'
)

mask = geo_feats['lat_cliente'].notna() & geo_feats['lat_vendedor'].notna()
geo_feats['distancia_km'] = np.nan
geo_feats.loc[mask, 'distancia_km'] = geo_feats[mask].apply(
    lambda r: haversine(r.lat_cliente, r.lng_cliente, r.lat_vendedor, r.lng_vendedor), axis=1
)

geo_feats['mesmo_estado'] = (geo_feats['customer_state'] == geo_feats['seller_state']).astype(int)

regioes = {
    'AC':'Norte','AM':'Norte','AP':'Norte','PA':'Norte','RO':'Norte','RR':'Norte','TO':'Norte',
    'AL':'Nordeste','BA':'Nordeste','CE':'Nordeste','MA':'Nordeste','PB':'Nordeste',
    'PE':'Nordeste','PI':'Nordeste','RN':'Nordeste','SE':'Nordeste',
    'DF':'Centro-Oeste','GO':'Centro-Oeste','MS':'Centro-Oeste','MT':'Centro-Oeste',
    'ES':'Sudeste','MG':'Sudeste','RJ':'Sudeste','SP':'Sudeste',
    'PR':'Sul','RS':'Sul','SC':'Sul'
}
geo_feats['regiao_cliente'] = geo_feats['customer_state'].map(regioes).fillna('desconhecido')
geo_feats['distancia_km']   = geo_feats['distancia_km'].fillna(geo_feats['distancia_km'].median())

# ── agrega pagamentos por pedido (pode ter mais de um por pedido) ──
feats_pag = payments.groupby('order_id').agg(
    tipo_pagamento      = ('payment_type',        lambda x: x.mode()[0]),
    qtd_parcelas        = ('payment_installments', 'max'),
    valor_pago_total    = ('payment_value',        'sum'),
    qtd_tipos_pagamento = ('payment_type',         'nunique'),
).reset_index()

# ── média histórica de entrega e avaliação por vendedor ──────
hist_vendedor = orders.merge(seller_por_pedido, on='order_id', how='left').groupby('seller_id').agg(
    media_dias_entrega_vendedor = ('dias_entrega', 'mean'),
    total_pedidos_vendedor      = ('order_id',     'count'),
).reset_index()
hist_vendedor['media_dias_entrega_vendedor'] = hist_vendedor['media_dias_entrega_vendedor'].round(2)

feats_seller_review = reviews.merge(
    orders[['order_id']], on='order_id', how='inner'
).merge(seller_por_pedido, on='order_id', how='left').groupby('seller_id').agg(
    media_avaliacao_vendedor = ('review_score', 'mean'),
).reset_index()

feats_hist = hist_vendedor.merge(feats_seller_review, on='seller_id', how='left')
feats_hist['media_avaliacao_vendedor'] = feats_hist['media_avaliacao_vendedor'].fillna(
    feats_hist['media_avaliacao_vendedor'].median()
)

# ── junta todos os grupos de features em um dataset final ────
df = orders[['order_id', 'customer_id', 'dias_entrega', 'hora_compra',
             'dia_semana_compra', 'mes_compra', 'eh_fim_de_semana',
             'horas_ate_aprovacao', 'horas_ate_transportadora']].copy()

df = df.merge(feats_itens, on='order_id', how='left')
df = df.merge(feats_prod,  on='order_id', how='left')
df = df.merge(geo_feats[['order_id', 'seller_id', 'customer_state', 'seller_state',
                          'mesmo_estado', 'distancia_km', 'regiao_cliente']],
              on='order_id', how='left')
df = df.merge(feats_pag,  on='order_id', how='left')
df = df.merge(feats_hist, on='seller_id', how='left')

# preenche nulos numéricos restantes com a mediana da coluna
for col in df.select_dtypes(include='number').columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# ── salva o dataset pronto para o modelo ─────────────────────
df.to_csv(PASTA + 'dataset_features.csv', index=False)

print(f"Linhas: {df.shape[0]:,} | Colunas: {df.shape[1]} | Nulos: {df.isnull().sum().sum()}")
# ── exibe resumo organizado no terminal ──────────────────────
print("\n" + "="*60)
print(" DATASET FINAL — RESUMO")
print("="*60)
print(f"  Linhas:          {df.shape[0]:,}")
print(f"  Colunas:         {df.shape[1]}")
print(f"  Nulos restantes: {df.isnull().sum().sum()}")
print(f"  Média do target: {df['dias_entrega'].mean():.1f} dias")
print(f"  Mediana:         {df['dias_entrega'].median():.1f} dias")
print("="*60)

print("\n COLUNAS CRIADAS:")
grupos = {
    "Pedido":    ['dias_entrega','hora_compra','dia_semana_compra','mes_compra','eh_fim_de_semana','horas_ate_aprovacao','horas_ate_transportadora'],
    "Itens":     ['qtd_itens','qtd_vendedores','valor_total_pedido','valor_total_frete','multiplos_vendedores','proporcao_frete'],
    "Produto":   ['peso_total_gramas','volume_total_cm3','media_qtd_fotos','categoria_produto','tem_descricao','peso_por_item'],
    "Geografia": ['customer_state','seller_state','mesmo_estado','distancia_km','regiao_cliente'],
    "Pagamento": ['tipo_pagamento','qtd_parcelas','valor_pago_total','qtd_tipos_pagamento'],
    "Vendedor":  ['media_dias_entrega_vendedor','total_pedidos_vendedor','media_avaliacao_vendedor'],
}
for grupo, cols in grupos.items():
    print(f"\n  [{grupo}]")
    for col in cols:
        print(f"    • {col}")

print("\n" + "="*60)
print(" AMOSTRA — 3 PRIMEIROS PEDIDOS")
print("="*60)
for i, row in df.head(3).iterrows():
    print(f"\n  Pedido {i+1}:")
    print(f"    dias_entrega:       {row['dias_entrega']:.1f} dias")
    print(f"    distancia_km:       {row['distancia_km']:.0f} km")
    print(f"    mesmo_estado:       {'sim' if row['mesmo_estado'] else 'nao'}")
    print(f"    tipo_pagamento:     {row['tipo_pagamento']}")
    print(f"    valor_total_pedido: R$ {row['valor_total_pedido']:.2f}")
    print(f"    media_vendedor:     {row['media_dias_entrega_vendedor']:.1f} dias")

print("\n" + "="*60)
print(f"  Salvo em: {PASTA}dataset_features.csv")
print("="*60 + "\n")