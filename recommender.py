import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


class AprioriRecommender:
    def __init__(self, ratings_path, copies_path, books_path):
        self.ratings_path = ratings_path
        self.copies_path = copies_path
        self.books_path = books_path
        self.rules = None
        self._prepare()

    def _prepare(self):
        # --- 1. Cargar y limpiar los datos ---
        ratings = pd.read_csv(self.ratings_path)
        copies = pd.read_csv(self.copies_path)
        books = pd.read_csv(self.books_path)

        # Eliminar nulos
        ratings.dropna(subset=['user_id', 'copy_id', 'rating'], inplace=True)
        copies.dropna(subset=['copy_id', 'book_id'], inplace=True)
        books.dropna(subset=['book_id', 'title'], inplace=True)

        # Eliminar valores fuera de rango y duplicados
        ratings = ratings[(ratings['rating'] >= 1) & (ratings['rating'] <= 5)]
        ratings.drop_duplicates(inplace=True)
        copies.drop_duplicates(inplace=True)
        books.drop_duplicates(subset='book_id', inplace=True)

        # --- 2. Unir ratings con libros ---
        ratings_books = ratings.merge(copies, on='copy_id', how='left')
        ratings_books = ratings_books.merge(books[['book_id', 'title']], on='book_id', how='left')

        # --- 3. Filtrar solo valoraciones positivas ---
        ratings_books = ratings_books[ratings_books['rating'] >= 4]

        # --- 4. Crear tabla usuario x libro ---
        basket = ratings_books.groupby(['user_id', 'title'])['rating'].count().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)

        # --- 5. Apriori ---
        frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

        # --- 6. Reglas de asociación ---
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        self.rules = rules.sort_values(by='lift', ascending=False)

    def recomendar_libros(self, libro_titulo, top_n=5):
        libro_titulo = libro_titulo.lower()
        recomendados = []

        for _, row in self.rules.iterrows():
            antecedents = [x.lower() for x in row['antecedents']]
            if libro_titulo in antecedents:
                recomendados.extend(row['consequents'])

        recomendados = list(set(recomendados))
        return recomendados[:top_n]
