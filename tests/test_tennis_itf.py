from analise.ingest.tennis_itf import parse_page, player_ids

HEAD = ('<tbody><tr class="js-tournament"><th class="h-text-left" colspan="2"><a href="/tennis/{cat}/x/" '
        'class="table-main__tournament table-main__tournament--tennis"><i><img src="a.svg"></i>'
        '<i class="tennis_tournament_type"></i>{title}</a></th></tr>')
FIN = ('<tr data-fro="1" data-dt="17,11,2026,13,00" data-def="1"> <td class="table-main__tt">'
       '<span class="table-main__time table-main__time--fin">FIN</span><a href="/tennis/itf-women-singles/'
       'w15-alcala/{slug}/plkwl1eR/" class="table-main__teamsLink"><span class="table-main__teamLine '
       'table-main__teamLine--home">{home}</span><span class="table-main__teamLine table-main__teamLine--away">'
       '{away}</span></a></td> <td class="table-main__result"><a href="/x/"> <strong>{res}</strong></a></td> '
       '<td class="table-main__partial" colspan="2">{part}</td> </tr>')
NEXT = ('<tr data-fro="0" data-dt="19,11,2026,0,00" data-def="1"> <td class="table-main__tt">'
        '<span class="table-main__time">00:00</span><a href="/tennis/itf-women-singles/w75-sydney/'
        'garland-joanna-broadus-savannah/614v5ZED/" class="table-main__teamsLink"><span class="table-main__teamLine '
        'table-main__teamLine--home">Garland J.</span><span class="table-main__teamLine table-main__teamLine--away">'
        'Broadus S.</span></a></td> <td class="table-main__streams"></td> <td class="table-main__odds">'
        '<button data-odd="1.09" data-odd-max="1.09"></button></td><td class="table-main__odds">'
        '<button data-odd="6.50"></button></td></tr>')


def test_player_ids_ordem_dos_nomes():
    assert player_ids("lee-carol-youngsuh-panshina-varvara", "Lee C. Y.", "Panshina V.") == \
        ("carol-lee-youngsuh", "panshina-varvara")
    # nome próprio primeiro numa e apelido primeiro na outra: o mesmo identificador
    a = player_ids("taraba-wallberg-nellie-kira-pavlova", "Taraba Wallberg N.", "Pavlova K.")
    b = player_ids("kira-pavlova-du-pree-britt", "Pavlova K.", "Du Pree B.")
    assert a == ("nellie-taraba-wallberg", "kira-pavlova") and b[0] == "kira-pavlova"
    assert player_ids("myatovic-julie-madelief-hageman", "Myatovic J.", "Hageman M.") == \
        ("julie-myatovic", "hageman-madelief")


def test_parse_resultados_e_proximos():
    page = (HEAD.format(cat="itf-women-singles", title="ITF Women - Singles: W15 Alcala, clay")
            + FIN.format(slug="gallardo-carmen-patier-rocio", home="<strong>Gallardo C.</strong>",
                         away="Patier R.", res="2:1", part="(6:3, 6<sup>4</sup>:7, 6:0)")
            + FIN.format(slug="toma-maria-makarova-mariia", home="Toma M.", away="<strong>Makarova M.</strong>",
                         res="0:1 RET.", part="(3:6, 1:2)")
            + FIN.format(slug="a-b-c-d", home="A B.", away="C D.", res="WO.", part="")
            + "</tbody>"
            + HEAD.format(cat="itf-men-singles", title="ITF Men - Singles: M15 X, hard")
            + FIN.format(slug="x-y-z-w", home="<strong>X Y.</strong>", away="Z W.", res="2:0", part="(6:0, 6:0)")
            + "</tbody>"
            + HEAD.format(cat="itf-women-singles", title="ITF Women - Singles: W75 Sydney, hard")
            + NEXT + "</tbody>")
    df = parse_page(page)
    assert list(df["status"]) == ["final", "desistencia", "por jogar"]  # sem masculino nem W.O.
    first = df.iloc[0]
    assert first["league"] == "ITF" and first["surface"] == "terra" and first["winner"] == 0
    assert (first["home_s2"], first["away_s2"]) == (6.0, 7.0) and first["home_sets"] == 2.0
    assert str(first["date"]) == "2026-11-17" and first["time"] == "12:00"  # UTC+1 -> Lisboa
    assert df.iloc[1]["winner"] == 1
    nxt = df.iloc[2]
    assert nxt["surface"] == "rapido" and (nxt["odds_home"], nxt["odds_away"]) == (1.09, 6.5)
    assert nxt["home_id"] == "garland-joanna"
