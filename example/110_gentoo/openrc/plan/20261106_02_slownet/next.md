  Prüfe das so:

    zcat /proc/config.gz | grep -E 'NET_SCH_INGRESS|NET_CLS_ACT|CLS_U32|NET_ACT_POLICE|NET_IFB'

  Wahrscheinlich fehlt NET_SCH_INGRESS. Dein Report nennt nur BBR und FQ. Ingress, U32 und Police stehen da nicht drin.

  Versuch noch die moderne Variante:

    sudo tc qdisc add dev wlan0 clsact

  Kommt wieder unknown, fehlt die Kernel-Unterstützung sicher.

  Lösung: in deiner Kernel-Config setzen:

    CONFIG_NET_SCH_INGRESS=y
    CONFIG_NET_CLS_ACT=y
    CONFIG_CLS_U32=y
    CONFIG_NET_ACT_POLICE=y

  Für sauberes Ingress-Shaping per ifb zusätzlich:

    CONFIG_NET_IFB=y
    CONFIG_NET_ACT_MIRRED=y
    CONFIG_CLS_MATCHALL=y

  Danach Image neu bauen, booten, dann erst:

    sudo tc qdisc add dev wlan0 handle ffff: ingress
    sudo tc filter add dev wlan0 parent ffff: protocol ip u32 match u32 0 0 police rate 48kbit burst 32kbit drop

  Ohne Neubau geht aktuell nur Senden drosseln. Empfangen drosseln braucht Ingress-Support.
