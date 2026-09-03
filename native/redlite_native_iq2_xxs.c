#include "redlite_native_iq2_xxs.h"

#include <stdio.h>
#include <string.h>

static const char kIQ2GridHex[] =
    "00000200050008000a00110014002000220028002a0041004400500058006100"
    "6400800082008a00a20001010401100115014001840198010002020222028202"
    "010404041004210424044004420448046004810484049004a404000502050805"
    "200546056905800591050906100640068406a406000805080808140828084108"
    "440850085208880804094009020a140a01100410101021104010601084109010"
    "951000110811201150115a118011241245120014081420142514491480141815"
    "6215001616160118041810184018811800190519a019511a002002200a204420"
    "6120802082202921482100220222012404241024402456240025412564259026"
    "082820289428442a014004401040184021402440404048405640604081408440"
    "9040004120416141804185410142104248425642684200440844204480449944"
    "124524450046014804481048404845480049584961498249454a904a00500850"
    "1150195020508050885004514251a4519152905492540a550156545600581158"
    "195864584059085a046010604060686000615561186260620064056410651265"
    "84654268008002800a8041808280048118814081118201840484108415844084"
    "608400854685948509864086608602880489118a0490109024904090a1901691"
    "8091459200942294449451958198209902a050a085a009a100a218a450a804a9";

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int rl_native_iq2_xxs_build_grid(uint8_t out[RL_IQ2_XXS_GRID_COUNT], char *error, size_t error_cap) {
    if (!out) {
        if (error && error_cap) snprintf(error, error_cap, "null IQ2_XXS grid output");
        return 0;
    }
    const size_t hex_len = strlen(kIQ2GridHex);
    if (hex_len != 1024u) {
        if (error && error_cap) snprintf(error, error_cap, "unexpected IQ2_XXS grid hex length %zu", hex_len);
        return 0;
    }
    static const uint8_t map[3] = {0x08u, 0x19u, 0x2bu};
    size_t out_pos = 0;
    for (size_t i = 0; i < hex_len; i += 2) {
        const int hi = hex_nibble(kIQ2GridHex[i]);
        const int lo = hex_nibble(kIQ2GridHex[i + 1]);
        if (hi < 0 || lo < 0) {
            if (error && error_cap) snprintf(error, error_cap, "invalid IQ2_XXS grid hex");
            return 0;
        }
        const uint8_t packed = (uint8_t)((hi << 4) | lo);
        for (uint32_t shift = 0; shift < 8u; shift += 2u) {
            const uint8_t code = (packed >> shift) & 3u;
            if (code >= 3u || out_pos >= RL_IQ2_XXS_GRID_COUNT) {
                if (error && error_cap) snprintf(error, error_cap, "invalid IQ2_XXS grid code");
                return 0;
            }
            out[out_pos++] = map[code];
        }
    }
    if (out_pos != RL_IQ2_XXS_GRID_COUNT) {
        if (error && error_cap) snprintf(error, error_cap, "IQ2_XXS grid expansion produced %zu bytes", out_pos);
        return 0;
    }
    if (error && error_cap) error[0] = '\0';
    return 1;
}
