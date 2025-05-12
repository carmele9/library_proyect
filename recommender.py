import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


class AprioriRecommender:
    def __init__(self, ratings_path, copies_path, books_path,
                 min_support=0.05, min_users=100, min_books_rated=3):
        self.ratings_path = ratings_path
        self.copies_path = copies_path
        self.books_path = books_path
        self.min_support = min_support
        self.min_users = min_users
        self.min_books_rated = min_books_rated
        self.rules = None

        self._load_and_prepare_data()
        self._apply_apriori()
        self._generate_rules()

    def _load_and_prepare_data(self):
        # Cargar datos
        ratings = pd.read_csv(self.ratings_path)
        copies = pd.read_csv(self.copies_path)
        books = pd.read_csv(self.books_path, quotechar='"', sep=',', encoding='utf-8', on_bad_lines='skip')

        # Limpiar datos
        ratings.dropna(subset=['user_id', 'copy_id', 'rating'], inplace=True)
        copies.dropna(subset=['copy_id', 'book_id'], inplace=True)
        books.dropna(subset=['book_id', 'title'], inplace=True)

        ratings = ratings[(ratings['rating'] >= 1) & (ratings['rating'] <= 5)]

        ratings.drop_duplicates(inplace=True)
        copies.drop_duplicates(inplace=True)
        books.drop_duplicates(subset='book_id', inplace=True)

        # Unir ratings con libros
        ratings_books = ratings.merge(copies, on='copy_id', how='left')
        ratings_books = ratings_books.merge(books[['book_id', 'title']], on='book_id', how='left')

        # Filtrar valoraciones positivas (4 o más)
        ratings_books = ratings_books[ratings_books['rating'] >= 4]

        # Filtrar libros populares
        libros_filtrados = ratings_books['title'].value_counts()
        libros_populares = libros_filtrados[libros_filtrados >= self.min_users].index

        # Filtrar usuarios activos
        user_activity = ratings_books['user_id'].value_counts()
        usuarios_activos = user_activity[user_activity >= self.min_books_rated].index

        # Aplicar filtros
        filtered_ratings = ratings_books[
            ratings_books['title'].isin(libros_populares) &
            ratings_books['user_id'].isin(usuarios_activos)
            ]

        # Crear cesta
        basket = filtered_ratings.groupby(['user_id', 'title'])['rating'].count().unstack().fillna(0)
        self.basket = basket > 0  # Convertir a booleano para Apriori

    def _apply_apriori(self):
        self.frequent_itemsets = apriori(self.basket, min_support=self.min_support, use_colnames=True)

    def _generate_rules(self):
        rules = association_rules(self.frequent_itemsets, metric="lift", min_threshold=1.0)
        self.rules = rules.sort_values(by='lift', ascending=False)

    def recomendar_libros(self, basado_en_libro, top_n=5):
        basado_en_libro = basado_en_libro.lower()
        recomendados = []

        for _, row in self.rules.iterrows():
            if basado_en_libro in [x.lower() for x in row['antecedents']]:
                recomendados.extend(row['consequents'])

        recomendados = list(set(recomendados))
        return recomendados[:top_n]


# Ejemplo de uso:
if __name__ == '__main__':
    recommender = AprioriRecommender(
        'casa_cultura_data/ratings.csv',
        'casa_cultura_data/copies(ejemplares).csv',
        'casa_cultura_data/books.csv',
        min_support=0.05,
        min_users=100,
        min_books_rated=3
    )

    print("Si te gustó 'The Hobbit', también podrías leer:")
    print(recommender.recomendar_libros('The Hobbit'))
