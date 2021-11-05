#!/usr/bin/python3
# coding: utf-8

import requests
from bs4 import BeautifulSoup
from scrapy import Selector
import csv
import datetime


def extract(url):
    """
    Export all cryptodata from cryptomarketcap.com
    website
    Arguments:
         url (str):
            url of the aimed Coinmarketcap page
    Return:
        .csv file
    """

    # INITIALISATION
    r = requests.session()
    start = datetime.datetime.now()

    # COLLECTE DU CODE SOURCE
    for retry in range(10):  # ajout d'un retrier en cas de site inaccessible
        response = r.get(url=url)
        print(response.headers)
        print("-- STATUS CODE --")
        print(response.status_code)

        # PARSING ET CREATION DU CSV
        if response.status_code == 200:
            with open("/path/to/coinmarketcap/cryptocurrencies_{}.csv".format(str(datetime.date.today())), "w") as f:
                fieldnames = ['Nom', 'Symbole', 'Cap. marché', 'Prix', 'Offre en circulation', 'Volume (24h)', '% 1h', '% 24h', '7 j']
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()

                soup = BeautifulSoup(response.text, features='html.parser')
                sel = Selector(text=soup.prettify())
                cryptos = sel.xpath("//tr[contains(@id, 'id-')]").extract()
                print(cryptos)
                for crypto in cryptos:
                    soup = BeautifulSoup(crypto, features='html.parser')
                    sel = Selector(text=soup.prettify())

                    nom = sel.xpath("//td[contains(@class, 'currency-name')]/@data-sort").extract_first()
                    symbole = sel.xpath("//td[contains(@class, 'col-symbol')]/text()").extract_first()
                    cap_marche = sel.xpath("//td[contains(@class, 'market-cap')]/text()").extract_first()
                    prix = sel.xpath("//a[@class='price']/@data-usd").extract_first()
                    offre_circulation = sel.xpath("//a[@class='volume']/@data-usd").extract_first()
                    volume = sel.xpath("//td[contains(@class, 'circulating-supply')]/@data-sort").extract_first()
                    percent_1h = sel.xpath("//td[@data-timespan='1h']/@data-sort").extract_first()
                    percent_24h = sel.xpath("//td[@data-timespan='24h']/@data-sort").extract_first()
                    percent_7j = sel.xpath("//td[@data-timespan='7d']/@data-sort").extract_first()

                    clean_values = []
                    values = [nom, symbole, cap_marche, prix, offre_circulation, volume, percent_1h, percent_24h, percent_7j]
                    for value in values:
                        if value:
                            value = value.strip().replace('\n', '')
                        clean_values.append(value)

                    print(', '.join(clean_values))
                    dict_row = dict(zip(fieldnames, clean_values))
                    writer.writerow(dict_row)

            # TEMPS PASSE
            end = datetime.datetime.now()
            time_elapsed = str(end - start)
            print('\n')
            print('-- TIME ELAPSED --')
            print(time_elapsed)
            break

        elif response.status_code == 404:
            print("Page indisponible")
            break

        else:
            print("Impossible d'accéder à la page")
            return []


def main():
    url = "https://coinmarketcap.com/fr/all/views/all/"
    extract(url)


if __name__ == '__main__':
    main()