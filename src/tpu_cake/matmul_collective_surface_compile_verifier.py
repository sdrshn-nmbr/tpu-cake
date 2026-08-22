from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import stat
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_EXECUTOR_SCHEMA = "matmul-collective-surface-compile-executor-v1"
_REPORT_SCHEMA = "matmul-collective-surface-compile-v1"
_DESIGN_SCHEMA = "matmul-collective-surface-design-v1"
_IDENTITY_SCHEMA = "length-prefixed-v2"
_COMPILER_ANALYSIS_SCHEMA = "compiler-executable-analysis-v2"
_FINAL_STATES = ("created", "verified", "lowered", "compiled")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_SOURCE_HASHES = {
    "tpu_cake/__init__.py": "fadd2bef5381a80e22ede32fca716303ecf6530c32d64776ebda4e498a699433",
    "tpu_cake/artifacts.py": "2c538cd0e4d8c6b50224670c1504a6e14d054c27d2b0aad4acb659464af561dc",
    "tpu_cake/canonical.py": "29d3f0165e1292901e49ebb8551b0b3ee32768b5ac3fca6d889df57c43f77cde",
    "tpu_cake/compiler_analysis.py": "12383974e0b8d090648820fe9c9dd9d861577d834a520eee0939126a2a1dc091",
    "tpu_cake/contracts.py": "3b3cfeb1e6b1036045d030db3b00967f3ad474ccd48ebf8a5c08bd0e3ef54e5b",
    "tpu_cake/cost_model.py": "816e11c0dcc3169dc03234368cfb16e98b497a3c8f08ece23bc0ef91ff482f7d",
    "tpu_cake/dialects/__init__.py": "fadd2bef5381a80e22ede32fca716303ecf6530c32d64776ebda4e498a699433",
    "tpu_cake/dialects/distributed_tensor.py": "de56d8206ba1c252da51ae0da89a3dfc0dbf009ab133eaea8cdf5d381c06d7a1",
    "tpu_cake/dialects/tpu_schedule.py": "0960681c7d12d10d75d265333b29906c539b1315d9f8082bdd80c637e2a6b929",
    "tpu_cake/distributed_frontend.py": "f17599291a00d06124b1fd5debd67ba938a1df798ed7f671bae3819032ba67a6",
    "tpu_cake/evidence.py": "ba7fd05982596a5d195327e25b2fabda9a628b2fddb6083222cc258d439e4196",
    "tpu_cake/frontend.py": "112904285ba197da3dee88b9a34f858397ce22334ff4a5ba00c301c1a07f1d48",
    "tpu_cake/identity.py": "ab9813b80a2f13e54494fcb045048edf047f5509d1a20cf5f971c71ce6cf78e8",
    "tpu_cake/ledger.py": "ddeff9309596d5248a1c7b90bf03ea26ee0d3afb93473c0d0b1bf84b4c21da06",
    "tpu_cake/lowering.py": "3615333950614198e0cbdb978342382440717bbfc1b291bcde6d7c4da503f64e",
    "tpu_cake/matmul_collective_surface_prediction.py": "32e94beca6e487bdc76dd7d3ea9932146cfb6d4f7e7c8105cbb0a5852ff5f382",
    "tpu_cake/matmul_collective_surface_runner.py": "cf7687d700742c3f4d50424e9a9364d8138df643efcf70f6dca7aeead3713d04",
    "tpu_cake/metrics.py": "a87f0bbc26f291e6bcec9932c356b866f529224042b0f673b434687d6666ccc0",
    "tpu_cake/pallas_lowering.py": "b40cd73db7cd4c12b6639064e5dd04b8e3b53be68eec72b34a5b3adace061ad5",
    "tpu_cake/receipt.py": "e2f2bf02d38e4af67570fb4bd209cfaf5ecc240d26a1b1d714b1745cc07fb32b",
    "tpu_cake/receipt_metrics.py": "7c57de6d8d3550337ca070853708da1d310208d547ec515d3ee78d409b862cb6",
    "tpu_cake/rpa_lowering.py": "902fab180b32181a24943e7f973c7f6e47baf9e5bb6af43b23636d004ed8dd7e",
    "tpu_cake/rpa_owned_kernel.py": "17b2c99bd4cd76fef29ed17e4e4dd9238a9a1442015083b3995aa251b6342e88",
    "tpu_cake/runner.py": "2af13a1dd60a2ef0e645367fd5afe825dc8f367d40043ecca0ec7957243c9c1b",
    "tpu_cake/search.py": "9aadd134e2723f1b63cef37ec021b0d4d15c105c6521ceaa17ebb52fa692a373",
    "tpu_cake/source.py": "783c764224811a4bee2e6f1e4bc100a2deaabfbb8442b61cdc9670faabcc188d",
    "tpu_cake/stablehlo.py": "cb123f8db96948e8bd9ae771cdbd5ca27afe9a56aa26a993c1cb9a60fa2410b0",
    "tpu_cake/workloads/__init__.py": "269acae151f69c0e59645bf79b367c98821d798678bedeb3d09a63c9eb17785e",
    "tpu_cake/workloads/distributed_matmul.py": "e5383acfd7e1c7416c798ec22211db6335e12cab89de3d96ef870f50affb26f2",
    "tpu_cake/workloads/inkling_rpa.py": "e5d0ccd6c79d66888ea7f6da1b9c97f009856cda4a2591b45e94a9fdb2cd8e83",
    "tpu_cake/workloads/matmul.py": "4e545bfa7866a7798fb98b3e9ee8e58cfb9643f5b04dfdf072a96b357dc9ccc6",
    "tpu_cake/xprof_evidence.py": "1357df464c1ab02e7f6ef4e7c256be72821172ed9b4011810465fb695122da9e",
}
_EXPECTED_SOURCE_DEPENDENCIES = tuple(_EXPECTED_SOURCE_HASHES)
_EXPECTED_UV_LOCK_SHA256 = "03c153a4daf4f1bf2c77d89620824e4f6c11fa946a9166f0f512e195d1025ed9"
_EXPECTED_EXECUTOR_SOURCE_SHA256 = (
    "ff56e0fe18a7c14cf83183fe088888d2ec587ff8965742c81d89011fa15fa00d"
)
_EXPECTED_WORKER_SOURCE_SHA256 = "986140711cd31df6baf3a97d6bdece579aceeef8b8493c11288d0f8eadf30eca"
_EXPECTED_DESIGN_ID = "f2f8a0eeba4842167780cd3d79043443d0d02392ed037a5250df1a2218691d83"
_EXPECTED_ARM_IDENTITIES = {
    ("calibration-0", "xla_reduce_scatter"): (
        "f2c111506ca8dd3452910374780297deebae99200f8c6fe1784f4b8d7d33d04d",
        "1e443268464a830a6d92f2bf7fd8e0f93a3919d93f42d70245fe31bf78146ea4",
        "55e0d5742d720fd241e48f4d67c73598fa159e693bdb5a129e47d8b943a41e47",
    ),
    ("calibration-0", "pallas_bidirectional_ring"): (
        "f2c111506ca8dd3452910374780297deebae99200f8c6fe1784f4b8d7d33d04d",
        "b9bd73fd5d63ca7af5cbe31d0d9f2e43cf3f8f01fc1a1d83e18f0bf96f48e1ca",
        "8ba158c44f8c5d8feec3374321453e601b250c99621f7b54554a30082ce185f9",
    ),
    ("calibration-1", "xla_reduce_scatter"): (
        "e79d8641deaffca19228cdba8d8d6ec633afaef005200331445456761a215de7",
        "34c4e5e768b57067c20b8e4d33435264d5ebc07d9acef15feb2625e236ac1f3d",
        "869a80f6e0ce47b39b0944332a7b9abcf87c9ebdad8c36c0476e8ee7980e8578",
    ),
    ("calibration-1", "pallas_bidirectional_ring"): (
        "e79d8641deaffca19228cdba8d8d6ec633afaef005200331445456761a215de7",
        "157ffa2b60ffe0e876470bad3b2228c8044d656cf818b52ce31a7ed82a37997b",
        "687806d6aa133fa3b4127ee580dcde69d2058b2ec3dc482b086b319124879fec",
    ),
    ("calibration-2", "xla_reduce_scatter"): (
        "426f35404557aa96c5e7635cfdf92347cc310f967c3ae407670186595837db9c",
        "873edb5e11d523f7fdf6e8dbb50280afe9e5b621d591b3dccfa9f89bbc41030b",
        "97d27caacefd7d5574f780348157e71d07c60b7124b4b927a06fb3c1566a4c7f",
    ),
    ("calibration-2", "pallas_bidirectional_ring"): (
        "426f35404557aa96c5e7635cfdf92347cc310f967c3ae407670186595837db9c",
        "77b67ff440bd9ed1e3b2c34b0ba7b3933bcefb32740781045992fcc44d8efb11",
        "f5237b474d19c4463d63be963ad4f8fe51c92fe9637c4b8de902e5fef2cbb55f",
    ),
    ("calibration-3", "xla_reduce_scatter"): (
        "429b07afdde5230c2a21cd94c8e6f42e24718eee0d17a0fc8fa08ba070eba922",
        "44eef1cf24b6bec66e2a4ebd3f220a9b65d78518030f58657ec594b8b3c6e65f",
        "faed3cd3022ddcd65692f613d96dee0612062351415abf7007cacf2445c6e823",
    ),
    ("calibration-3", "pallas_bidirectional_ring"): (
        "429b07afdde5230c2a21cd94c8e6f42e24718eee0d17a0fc8fa08ba070eba922",
        "bbce454881dc6700417b8c21eab4fc5c7222aa3fee09837692481feeebec24f5",
        "51f9f44805dce9e73a01c922b98019538914e5b2838760d8f8fa9acfe60db589",
    ),
    ("calibration-4", "xla_reduce_scatter"): (
        "94f407e58c802524cda2aaef5dc9343a442ebf209003204e38b971c6be3561f5",
        "5823b1cbc74512d92c3c73e02c867b49cf760f9f208774706a095b61c10ceca9",
        "4bb13c797d869e9e65921b01119f31fea977bfaec55af09f92dad998d0453cc8",
    ),
    ("calibration-4", "pallas_bidirectional_ring"): (
        "94f407e58c802524cda2aaef5dc9343a442ebf209003204e38b971c6be3561f5",
        "2e635dfcfad812663dd601f7e1050c8d9c219342fde9026d93f3e10174b78913",
        "266a9685d05182299ede798304c6473ea8bafaf46044684fffa30c012b20fc58",
    ),
    ("calibration-5", "xla_reduce_scatter"): (
        "cd35754866cc287b322e8369c1f0aa148e90fc54879b112df886ef5c2d006e8e",
        "b115433978a4ab15156751ee2a64e20043245b0c155a0e49c0b8f35e435da3df",
        "0acbcb88968f55345fac9aa5b6d3e9ac9791d07516b5319c149b470bede3e917",
    ),
    ("calibration-5", "pallas_bidirectional_ring"): (
        "cd35754866cc287b322e8369c1f0aa148e90fc54879b112df886ef5c2d006e8e",
        "f5a00b4f4c9084330ca6d2a15a767deace4203eb252da776a8b6103db7f2c420",
        "b2890ded5e4f338fffe9affed44055a8bc025eab7f255d6eab53ede7f5ef9a5e",
    ),
    ("calibration-6", "xla_reduce_scatter"): (
        "8d170633e2b0e7f68b98564581a9399421a16f25ec1bf2b20fba2fd80c678ad8",
        "9c1b5f0c916c84f77aa07bfd7cad708ba60106f0919fb1dd2c3860e983bb7dab",
        "cca088dade2c71df5a73121a9d140a440957bd2227bf747febce88e7eb3a6e12",
    ),
    ("calibration-6", "pallas_bidirectional_ring"): (
        "8d170633e2b0e7f68b98564581a9399421a16f25ec1bf2b20fba2fd80c678ad8",
        "49b1e77a3d43c5f637dc8b354d2a050a4daf2335d701a765763758f5adb6353d",
        "be172fb541edbdbe2ae7ab8ca803634eb3c109d65f6b656200ae894f3e2228af",
    ),
    ("calibration-7", "xla_reduce_scatter"): (
        "3d5a6c600c94d8a4625936d3b5b83c9501441a13d02fb6410df30fb7aea80869",
        "00503d9579b2936ef20e93095651a26713479607982ea221f793c9c9bcb6c1e3",
        "d4445fa20dca590179f1183e6b63ba5d1ac5d34caf10de7eed31a82b2f87ccf8",
    ),
    ("calibration-7", "pallas_bidirectional_ring"): (
        "3d5a6c600c94d8a4625936d3b5b83c9501441a13d02fb6410df30fb7aea80869",
        "57a144cd3c5561a4d122d42509543c5aa560704b411c6b43b69d2f201564df30",
        "3bf9ccdafda670307381a1a54d1e411da8617cbaa208c8f343c67f3ddb335529",
    ),
    ("calibration-8", "xla_reduce_scatter"): (
        "6d020a485a33daef7ef9da711da1f6960898c94acef1712ada177300b803f7c2",
        "55b71e1d35dda917afcfd49b969d684190ef9773594f51fb701b4ce52a2d4cfe",
        "4a8ceac0b9989f714f8b9f61a656c180234e79d37d6c57d109dca57e8245bf80",
    ),
    ("calibration-8", "pallas_bidirectional_ring"): (
        "6d020a485a33daef7ef9da711da1f6960898c94acef1712ada177300b803f7c2",
        "34a9dd4718b22540632afb2ba1189dd1be9bdcba7b58fc54dcee5e90685f7654",
        "a8c8cda06e7585a13cde58e6b973b0edb53f1c4a5c28ea6b4836d20e7a33866b",
    ),
    ("calibration-9", "xla_reduce_scatter"): (
        "5d9eb242b33a10958fa77b2661d787534e90fd6d005417691b3031c91d91bdf8",
        "b4a249badf81aab5388aa3b1cefa7d20280fdbe8d911ce29a3fcb727e18ca7bc",
        "1ac4543a07bb497f4a1b94e3b8bf31e0347b8f8be5bbd317f3ea5dd4ecd00e01",
    ),
    ("calibration-9", "pallas_bidirectional_ring"): (
        "5d9eb242b33a10958fa77b2661d787534e90fd6d005417691b3031c91d91bdf8",
        "bc917dcb9e7ca68a8ae471bcccaeb992b6b64fed83cebb625c29a3069b81a043",
        "11f0de80c9453fa61bb282ec5c7306758e79734c12fa5af5d7083be55a2ae248",
    ),
    ("calibration-10", "xla_reduce_scatter"): (
        "859e9f113fb24bcfea1e4cb4678e370882ebbfc5315d4b8a45a879f2470908e5",
        "ae68070fd83425cd6d7ebe6f67bce22e9960c9dc3bf3b3320dff4b50707c9817",
        "5b6e6e60a11e3e14f5847fe0124e27b357d4f7f9d802260096cbd82b7409c671",
    ),
    ("calibration-10", "pallas_bidirectional_ring"): (
        "859e9f113fb24bcfea1e4cb4678e370882ebbfc5315d4b8a45a879f2470908e5",
        "b615aaf744f43c93cde8bfe31e848e3523ebe977827fb1cb3aabd5918c31b188",
        "f800d2f77660627b797f2eb4941e11dc25389f192cc259156828e76fc80a91a3",
    ),
    ("calibration-11", "xla_reduce_scatter"): (
        "50802641038e124e61dc79ae1ed577b11cfdd9b3f1d885f9a99f48a7d8cba1cb",
        "0d5783606e6db12e084e2f630c671ec69434387bb6f39b9db859c8dec5af5431",
        "22c19ed144278f268f26d739e0917541dcb27a0fe82ea0085c04061542cc1c44",
    ),
    ("calibration-11", "pallas_bidirectional_ring"): (
        "50802641038e124e61dc79ae1ed577b11cfdd9b3f1d885f9a99f48a7d8cba1cb",
        "a9f78786474a1fedd8bf437da60e3a8ad9495796436ce70f0d133f3fae5dc478",
        "56117c99fde0c93536738e7aa4912798bf51c1ff866549a3986a036581e71028",
    ),
    ("calibration-12", "xla_reduce_scatter"): (
        "7952bc7a760c0210b73a6021b47754bab87cc9df6373edafd42719b3d2d760a0",
        "f326f41847bafe11d912d664d9214b6d0e87e2ef74c81078883eeeeaff719127",
        "3473595d707e2e91230dddc838457dcdd0507791d4e074bd6a83ff4eb136fe9e",
    ),
    ("calibration-12", "pallas_bidirectional_ring"): (
        "7952bc7a760c0210b73a6021b47754bab87cc9df6373edafd42719b3d2d760a0",
        "9de8729583deb2ac4189ffbaea597e12a8a6aaa6ad89aa38851bfa9b77829e04",
        "e793dc4bc559bd744a94f7c732fc7025bc5e455d3d2f976eaaa662c3cb3262c1",
    ),
    ("calibration-13", "xla_reduce_scatter"): (
        "ca7d6afb0d51fe088b187b68a0964fdbc0aef666092952e4cf085c55eaead70e",
        "c0a8034403d93f17eb5e0469875329d2ef3488a4a7222ddcc9963918cae2a6ad",
        "835fdf051b5f307d19036b84e6fec94dbe35076ac6aa0f80e2fdbdeb6e6eb15c",
    ),
    ("calibration-13", "pallas_bidirectional_ring"): (
        "ca7d6afb0d51fe088b187b68a0964fdbc0aef666092952e4cf085c55eaead70e",
        "3cd5ee5ede7bd9d5d5105e675ec56e3e021505f2c18bfb2ad3f450cc16e3a92e",
        "89259ded3f896a506d504b96323607313f1d4470ab0e89d1178553250b8bc828",
    ),
    ("calibration-14", "xla_reduce_scatter"): (
        "7975a3ac702a8dcb2f98e17f2ea2345820e93b10a2ddc145bedb8c6a39111a3f",
        "340114145f982c6b0f9dd77fc238905393e2ac9542504844bf6fb84bbbc862e8",
        "f1754fa08479ce1e72c134a2cf50e06e71f333abacbac793bc3c0d609e040a08",
    ),
    ("calibration-14", "pallas_bidirectional_ring"): (
        "7975a3ac702a8dcb2f98e17f2ea2345820e93b10a2ddc145bedb8c6a39111a3f",
        "28b4256c98917dfcd3fdb4d3e021ae22fbc9e8a7bd2ffd167ff61d54309e25be",
        "4f5a647312f2567db0b799e30983f7ef418ca374fae842afb865c66a97e76b8f",
    ),
    ("calibration-15", "xla_reduce_scatter"): (
        "88f827c9d43fbe9f3d66ad371115aa77f0046280a7084c236fd5c3b60e8231cc",
        "8dd430611e5a99edd38e4cb1b0ef3b2989f746e5e8a31a197aff396d02caa858",
        "f22085ca9fd748848ce205b6a6ec96dfedbdd2e8b2077a22a82301ae7f59184f",
    ),
    ("calibration-15", "pallas_bidirectional_ring"): (
        "88f827c9d43fbe9f3d66ad371115aa77f0046280a7084c236fd5c3b60e8231cc",
        "f27c4bc5a5573b742654346bd779dac3778c49d3f7f0eeacce93d3e2ad3e31ed",
        "452c2f5330008b7de4a25f1d5b641c8ecedb8c1a2f656c22820a2bf1bfd797f0",
    ),
    ("holdout-0", "xla_reduce_scatter"): (
        "4176c322a7cb5b11d1aba71836a23024774168f5de2a860c1b21826b17368095",
        "98826ce11c7624c7fe81b1d426926b7594597ca6e6de3d2612e32edd11158b27",
        "81c7d5b0cf217744656f75dc354b472598f49e851b955cbb5b8f1f077df9a4d7",
    ),
    ("holdout-0", "pallas_bidirectional_ring"): (
        "4176c322a7cb5b11d1aba71836a23024774168f5de2a860c1b21826b17368095",
        "23be168618ed358fb8da363ce7b9e3b0b8f74cd65327dcae6778cab96c2f4152",
        "9f2223634d956f45fee7f950332bc2aebbb116b0772fdb67f6bf56644d843c82",
    ),
    ("holdout-1", "xla_reduce_scatter"): (
        "2c501d10353aee1dff4bf554f40e9b226623128042647c4138282a3ce22b4c04",
        "a1cef633114ea1e74263b0e48cfa58459b759f7899b5504c171fdc7cd2bec753",
        "4cb8597ba96809c625dc5229b3b58cb311c9e8decd94895dccdcc48d2d935558",
    ),
    ("holdout-1", "pallas_bidirectional_ring"): (
        "2c501d10353aee1dff4bf554f40e9b226623128042647c4138282a3ce22b4c04",
        "0d309a6459318965f23f72141d5711ee9b7a5d1fc9fa81313d7f234dfcdcbe42",
        "982a7b3746c3f365aeab9214265a929e51466fabf6e3ebb749586ca38b24a3f6",
    ),
    ("holdout-2", "xla_reduce_scatter"): (
        "677623023e11eece7c34ea40bb8539bcfdbad4e35f0d5f46a4c4761e159b1cc2",
        "96651df764872835687d93cf14faa823761c451e62bfb012d9079442171ec819",
        "7c5fb380bd49b707c6d98264fc81b4433c889db986125de31bb70d654d38a5e1",
    ),
    ("holdout-2", "pallas_bidirectional_ring"): (
        "677623023e11eece7c34ea40bb8539bcfdbad4e35f0d5f46a4c4761e159b1cc2",
        "fccc54202d9adfbd858887bf196b20275537c3ca16f95813f201c524d3c4285d",
        "1f794e4260d84b749810f1215a7fbbb7281fa2d20d4b1e6131380676d73c73fa",
    ),
    ("holdout-3", "xla_reduce_scatter"): (
        "e1c74df68962b3ef49a0172b09aa00dbfa0b74f05ab970cc5e21ab07fe8b1385",
        "380b0d6803fc7ee36e280d936b4732a74274a50afb162fe00406370145346cdb",
        "7df5d166ff24c9390b9c301977eddea9ac1a026afcbcfd9240d7b88eb1196cc3",
    ),
    ("holdout-3", "pallas_bidirectional_ring"): (
        "e1c74df68962b3ef49a0172b09aa00dbfa0b74f05ab970cc5e21ab07fe8b1385",
        "99a7aa7ee14abbfc53d8f0b3f37867cbed00d7aa085f88d2bd742ea0cf6e8d0f",
        "909f67d7b8b09312fda7a3064a19a58df11ef6dd82a61ea6a54f892d2316de82",
    ),
}

_CAPTURE_KEYS = {
    "scenario_name",
    "strategy",
    "repetition",
    "input_contract_sha256",
    "distributed_schedule_sha256",
    "physical_schedule_sha256",
    "pallas_source_sha256",
    "status",
    "stablehlo",
    "compiler_hlo",
    "stablehlo_sha256",
    "semantic_stablehlo_sha256",
    "compiler_hlo_sha256",
    "semantic_compiler_hlo_sha256",
    "error_sha256",
}
_ABI_KEYS = {
    "lhs_shape",
    "lhs_dtype",
    "lhs_sharding",
    "rhs_shape",
    "rhs_dtype",
    "rhs_sharding",
    "output_shape",
    "output_dtype",
    "output_sharding",
    "schema_version",
}
_ANALYSIS_MEMORY_KEYS = {
    "generated_code_size_in_bytes",
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "host_generated_code_size_in_bytes",
    "host_argument_size_in_bytes",
    "host_output_size_in_bytes",
    "host_alias_size_in_bytes",
    "host_temp_size_in_bytes",
    "peak_memory_in_bytes",
    "buffer_assignment_available",
    "buffer_assignment_size_bytes",
    "buffer_assignment_sha256",
}
_COLLECTIVE_KEYS = {
    "stablehlo_reduce_scatter_count",
    "stablehlo_all_gather_count",
    "compiler_reduce_scatter_count",
    "compiler_all_reduce_count",
    "compiler_all_gather_count",
    "sparse_core_reduce_scatter_count",
    "sparse_core_all_gather_count",
}


@dataclass(frozen=True)
class VerifiedCompileCapture:
    repetition: int
    scenario_name: str
    strategy: str
    abstract_input_abi_sha256: str
    stablehlo_path: str
    stablehlo_sha256: str
    compiler_hlo_path: str
    compiler_hlo_sha256: str
    compiler_analysis_path: str
    compiler_analysis_sha256: str


@dataclass(frozen=True)
class SurfaceCompileVerification:
    attempt_id: str
    design_id: str
    source_authority_sha256: str
    execution_authority_sha256: str
    compile_report_sha256: str
    ledger_sha256: str
    ledger_states: tuple[str, ...]
    verifier_source_sha256: str
    verifier_canonically_bound: bool
    captures: tuple[VerifiedCompileCapture, ...]

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _reject_constant(value: str) -> None:
    raise ValueError(f"SURFACE_COMPILE_INDEPENDENT_JSON_CONSTANT value={value}")


def _pairs_to_dict(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"SURFACE_COMPILE_INDEPENDENT_JSON_DUPLICATE_KEY key={key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_pairs_to_dict,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"SURFACE_COMPILE_INDEPENDENT_JSON_INVALID path={path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"SURFACE_COMPILE_INDEPENDENT_JSON_OBJECT_REQUIRED path={path}")
    return value


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"SURFACE_COMPILE_INDEPENDENT_{label}_SCHEMA_MISMATCH")


def _require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise TypeError(f"SURFACE_COMPILE_INDEPENDENT_{label}_INVALID")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"SURFACE_COMPILE_INDEPENDENT_{label}_INVALID")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _identity_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _semantic_sha256(*parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"tpu-cake-semantic-identity\x00length-prefixed-v2\x00")
    for part in parts:
        if not part:
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_SEMANTIC_IDENTITY_EMPTY")
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _canonical_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"SURFACE_COMPILE_INDEPENDENT_{label}_INVALID")
    path = PurePosixPath(value)
    if path.is_absolute() or not value or value != path.as_posix() or ".." in path.parts:
        raise ValueError(f"SURFACE_COMPILE_INDEPENDENT_{label}_INVALID")
    return value


def _validate_archive_tree(root: Path) -> None:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_ROOT_INVALID")
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_ROOT_INVALID")
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_ARCHIVE_LINK")
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_ARCHIVE_HARDLINK")
        if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_ARCHIVE_FILE_TYPE")


def _validate_manifest(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _read_json(root / "manifest.json")
    _expect_keys(
        manifest,
        {"schema_version", "identity", "report_sha256", "ledger_sha256", "artifacts"},
        "MANIFEST",
    )
    if manifest["schema_version"] != _EXECUTOR_SCHEMA:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_MANIFEST_SCHEMA_MISMATCH")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_MANIFEST_ARTIFACTS_INVALID")
    by_path: dict[str, dict[str, Any]] = {}
    previous = ""
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise TypeError("SURFACE_COMPILE_INDEPENDENT_MANIFEST_ENTRY_INVALID")
        _expect_keys(entry, {"path", "size", "sha256"}, "MANIFEST_ENTRY")
        relative = _canonical_relative_path(entry["path"], "MANIFEST_PATH")
        if relative <= previous or relative == "manifest.json":
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_MANIFEST_ORDER_MISMATCH")
        previous = relative
        size = _require_int(entry["size"], "MANIFEST_SIZE")
        digest = _require_hex(entry["sha256"], _HEX_64, "MANIFEST_HASH")
        path = root / relative
        if not path.is_file() or path.stat().st_size != size or _file_sha256(path) != digest:
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_MANIFEST_HASH_MISMATCH")
        by_path[relative] = entry
    observed_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed_files != {"manifest.json", *by_path}:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_MANIFEST_INVENTORY_MISMATCH")
    expected_directories = {
        parent.as_posix()
        for relative in by_path
        for parent in PurePosixPath(relative).parents
        if parent != PurePosixPath(".")
    }
    observed_directories = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    }
    if observed_directories != expected_directories:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_DIRECTORY_INVENTORY_MISMATCH")
    return manifest, by_path


def _validate_contract(contract_path: Path, root: Path) -> tuple[dict[str, Any], str]:
    supplied = _read_json(contract_path)
    recorded = _read_json(root / "contract.json")
    if supplied != recorded:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_CONTRACT_MISMATCH")
    if (
        recorded.get("schema_version") != _DESIGN_SCHEMA
        or recorded.get("identity_schema") != _IDENTITY_SCHEMA
        or recorded.get("compile_input_abi_schema") != "global-shape-dtype-named-sharding-v1"
        or recorded.get("compiler_analysis_schema") != _COMPILER_ANALYSIS_SCHEMA
        or recorded.get("compiler_capture_repetitions") != 2
        or recorded.get("require_stable_compiler_semantic_hashes") is not True
        or recorded.get("allow_retry") is not False
        or recorded.get("compile_duration_available_to_fit") is not False
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_CONTRACT_PROTOCOL_MISMATCH")
    strategies = recorded.get("strategies")
    scenarios = recorded.get("scenarios")
    if (
        strategies != ["xla_reduce_scatter", "pallas_bidirectional_ring"]
        or not isinstance(scenarios, list)
        or len(scenarios) != 20
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_CONTRACT_INVENTORY_MISMATCH")
    names: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise TypeError("SURFACE_COMPILE_INDEPENDENT_SCENARIO_INVALID")
        _expect_keys(scenario, {"name", "split", "m", "k", "n", "tile_m", "tile_n"}, "SCENARIO")
        name = scenario["name"]
        if not isinstance(name, str) or name in names:
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_SCENARIO_NAME_INVALID")
        for dimension in ("m", "k", "n", "tile_m", "tile_n"):
            _require_int(scenario[dimension], f"SCENARIO_{dimension.upper()}", minimum=1)
        names.append(name)
    design_id = _identity_sha256(recorded)
    if design_id != _EXPECTED_DESIGN_ID:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_CANONICAL_DESIGN_MISMATCH")
    return recorded, design_id


def _validate_source_and_execution_authority(
    root: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    authority = _read_json(root / "execution_authority.json")
    _expect_keys(
        authority,
        {
            "schema_version",
            "source",
            "executor_source_sha256",
            "worker_source_sha256",
            "verifier_source_sha256",
            "project",
            "zone",
            "hostname",
            "numeric_project_id",
            "instance_id",
            "instance_hostname",
            "machine_type",
            "cpu_platform",
            "backend",
            "runtime",
            "compiler_environment",
            "devices",
        },
        "EXECUTION_AUTHORITY",
    )
    if authority["schema_version"] != _EXECUTOR_SCHEMA:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_EXECUTION_SCHEMA_MISMATCH")
    source = authority["source"]
    if not isinstance(source, dict):
        raise TypeError("SURFACE_COMPILE_INDEPENDENT_SOURCE_AUTHORITY_INVALID")
    _expect_keys(
        source,
        {
            "source_commit",
            "branch",
            "origin_main_commit",
            "remote_main_commit",
            "remote_url",
            "compilation_source_root",
            "runtime",
            "uv_lock_sha256",
            "dependencies",
        },
        "SOURCE_AUTHORITY",
    )
    commit = _require_hex(source["source_commit"], _HEX_40, "SOURCE_COMMIT")
    if (
        source["origin_main_commit"] != commit
        or source["remote_main_commit"] != commit
        or source["branch"] != contract.get("source_branch")
        or source["remote_url"] != contract.get("source_remote_url")
        or source["compilation_source_root"] != contract.get("compilation_source_root")
        or source["runtime"] != contract.get("runtime")
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_SOURCE_AUTHORITY_MISMATCH")
    contract_authority_fields = (
        "project",
        "zone",
        "hostname",
        "numeric_project_id",
        "instance_id",
        "instance_hostname",
        "machine_type",
        "cpu_platform",
        "backend",
        "runtime",
        "compiler_environment",
    )
    if any(authority[field] != contract.get(field) for field in contract_authority_fields):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_EXECUTION_AUTHORITY_MISMATCH")
    devices = authority["devices"]
    if not isinstance(devices, list) or len(devices) != contract.get("device_count"):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_DEVICE_INVENTORY_MISMATCH")
    device_ids: list[int] = []
    for device in devices:
        if not isinstance(device, dict):
            raise TypeError("SURFACE_COMPILE_INDEPENDENT_DEVICE_INVALID")
        _expect_keys(device, {"id", "process_index", "platform", "device_kind"}, "DEVICE")
        device_id = _require_int(device["id"], "DEVICE_ID")
        _require_int(device["process_index"], "DEVICE_PROCESS_INDEX")
        if (
            device["process_index"] != contract.get("device_process_index")
            or device["platform"] != contract.get("backend")
            or device["device_kind"] != contract.get("device_kind")
        ):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_DEVICE_AUTHORITY_MISMATCH")
        device_ids.append(device_id)
    if device_ids != contract.get("device_ids"):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_DEVICE_AUTHORITY_MISMATCH")
    dependencies = source["dependencies"]
    if not isinstance(dependencies, list):
        raise TypeError("SURFACE_COMPILE_INDEPENDENT_SOURCE_DEPENDENCIES_INVALID")
    dependency_paths: list[str] = []
    dependency_hashes: dict[str, str] = {}
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise TypeError("SURFACE_COMPILE_INDEPENDENT_SOURCE_DEPENDENCY_INVALID")
        _expect_keys(dependency, {"path", "sha256"}, "SOURCE_DEPENDENCY")
        path = _canonical_relative_path(dependency["path"], "SOURCE_DEPENDENCY_PATH")
        dependency_paths.append(path)
        dependency_hashes[path] = _require_hex(
            dependency["sha256"], _HEX_64, "SOURCE_DEPENDENCY_HASH"
        )
    if tuple(dependency_paths) != _EXPECTED_SOURCE_DEPENDENCIES:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_SOURCE_DEPENDENCY_CLOSURE_MISMATCH")
    if dependency_hashes != _EXPECTED_SOURCE_HASHES:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_CANONICAL_SOURCE_HASH_MISMATCH")
    bundle_root = root / "source" / "committed"
    expected_bundle = {*_EXPECTED_SOURCE_DEPENDENCIES, "uv.lock"}
    observed_bundle = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    if observed_bundle != expected_bundle:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_SOURCE_BUNDLE_INVENTORY_MISMATCH")
    for path, digest in dependency_hashes.items():
        if _file_sha256(bundle_root / path) != digest:
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_SOURCE_BUNDLE_HASH_MISMATCH")
    uv_hash = _require_hex(source["uv_lock_sha256"], _HEX_64, "UV_LOCK_HASH")
    if uv_hash != _EXPECTED_UV_LOCK_SHA256:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_CANONICAL_UV_LOCK_HASH_MISMATCH")
    if _file_sha256(bundle_root / "uv.lock") != uv_hash:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_UV_LOCK_HASH_MISMATCH")
    executor_hash = _require_hex(
        authority["executor_source_sha256"], _HEX_64, "EXECUTOR_SOURCE_HASH"
    )
    worker_hash = _require_hex(authority["worker_source_sha256"], _HEX_64, "WORKER_SOURCE_HASH")
    verifier_hash = _require_hex(
        authority["verifier_source_sha256"], _HEX_64, "VERIFIER_SOURCE_HASH"
    )
    if (
        executor_hash != _EXPECTED_EXECUTOR_SOURCE_SHA256
        or worker_hash != _EXPECTED_WORKER_SOURCE_SHA256
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_CANONICAL_OPERATIONAL_SOURCE_MISMATCH")
    if (
        _file_sha256(root / "source" / "executor.py") != executor_hash
        or _file_sha256(root / "source" / "worker.py") != worker_hash
        or _file_sha256(root / "source" / "verifier.py") != verifier_hash
        or _file_sha256(Path(__file__)) != verifier_hash
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_OPERATIONAL_SOURCE_HASH_MISMATCH")
    return authority, _identity_sha256(source), _identity_sha256(authority)


def _validate_run_identity(
    root: Path,
    contract: dict[str, Any],
    design_id: str,
    execution_authority_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    identity = _read_json(root / "run_identity.json")
    _expect_keys(
        identity,
        {
            "attempt_id",
            "design_id",
            "execution_authority_sha256",
            "producer_output_root",
            "attempt_claim_path",
            "attempt_claim_sha256",
        },
        "RUN_IDENTITY",
    )
    attempt_id = _require_hex(identity["attempt_id"], _HEX_64, "ATTEMPT_ID")
    producer_root = identity["producer_output_root"]
    if not isinstance(producer_root, str) or not Path(producer_root).is_absolute():
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_PRODUCER_ROOT_INVALID")
    claim_key = _sha256_bytes(f"{design_id}:{source_commit}".encode())
    expected_claim_path = str(Path(str(contract["attempt_registry_root"])) / f"{claim_key}.json")
    claim_payload = {
        "schema_version": _EXECUTOR_SCHEMA,
        "attempt_id": attempt_id,
        "design_id": design_id,
        "source_commit": source_commit,
        "output_root": producer_root,
        "state": "claimed",
    }
    if (
        identity["design_id"] != design_id
        or identity["execution_authority_sha256"] != execution_authority_sha256
        or identity["attempt_claim_path"] != expected_claim_path
        or identity["attempt_claim_sha256"] != _sha256_bytes(_pretty_json_bytes(claim_payload))
        or _read_json(root / "attempt.json") != {"attempt_id": attempt_id}
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_ATTEMPT_IDENTITY_MISMATCH")
    return identity


def _semantic_compiler_hlo(value: str) -> str:
    canonical = value.rstrip("\n") + "\n"
    metadata_start = canonical.find("\nFileNames\n")
    if metadata_start >= 0:
        starts = tuple(
            offset
            for marker in ("\n%", "\nENTRY ")
            if (offset := canonical.find(marker, metadata_start)) >= 0
        )
        if not starts:
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_COMPILER_HLO_INVALID")
        canonical = canonical[:metadata_start] + canonical[min(starts) :]
    return re.sub(r" stack_frame_id=\d+", "", canonical)


def _validate_static_abi(stablehlo: str, compiler_hlo: str, scenario: dict[str, Any]) -> None:
    uncommented = "\n".join(
        line for line in stablehlo.splitlines() if not line.lstrip().startswith("//")
    )
    signatures = re.findall(
        r"func\.func\s+public\s+@main\s*\((.*?)\)\s*->\s*\(?\s*(tensor<[^>]+>)",
        uncommented,
        flags=re.DOTALL,
    )
    expected_stable = (
        f"tensor<{scenario['m']}x{scenario['k']}xbf16>",
        f"tensor<{scenario['k']}x{scenario['n']}xbf16>",
        f"tensor<{scenario['m']}x{scenario['n']}xf32>",
    )
    if len(signatures) != 1:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_STABLEHLO_ABI_MISMATCH")
    parameters, result = signatures[0]
    if (*re.findall(r"tensor<[^>]+>", parameters), result) != expected_stable:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_STABLEHLO_ABI_MISMATCH")
    headers = tuple(
        line.strip() for line in compiler_hlo.splitlines() if line.strip().startswith("HloModule ")
    )
    if len(headers) != 1 or "entry_computation_layout=" not in headers[0]:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_COMPILER_HLO_ABI_MISMATCH")
    observed = tuple(
        (dtype, tuple(int(dimension) for dimension in dimensions.split(",") if dimension))
        for dtype, dimensions in re.findall(r"\b(bf16|f32)\[([0-9,]*)\]", headers[0])
    )
    expected_compiler = (
        ("bf16", (scenario["m"], scenario["k"])),
        ("bf16", (scenario["k"], scenario["n"])),
        ("f32", (scenario["m"], scenario["n"])),
    )
    if observed != expected_compiler:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_COMPILER_HLO_ABI_MISMATCH")


def _operation_count(value: str, operation: str) -> int:
    pattern = re.compile(rf"=\s*[^\n=]*\b{re.escape(operation)}\(")
    return sum(bool(pattern.search(line)) for line in value.splitlines())


def _stable_operation_count(value: str, operation: str) -> int:
    pattern = re.compile(rf'=\s*"?stablehlo\.{re.escape(operation)}"?(?=[\s(<])')
    return sum(bool(pattern.search(line)) for line in value.splitlines())


def _collectives(stablehlo: str, compiler_hlo: str) -> dict[str, int]:
    reduce_scatter_lines = tuple(
        line
        for line in compiler_hlo.splitlines()
        if re.search(r"=\s*[^\n=]*\breduce\-scatter\(", line)
    )
    all_gather_lines = tuple(
        line for line in compiler_hlo.splitlines() if re.search(r"=\s*[^\n=]*\ball\-gather\(", line)
    )
    return {
        "stablehlo_reduce_scatter_count": _stable_operation_count(stablehlo, "reduce_scatter"),
        "stablehlo_all_gather_count": _stable_operation_count(stablehlo, "all_gather"),
        "compiler_reduce_scatter_count": len(reduce_scatter_lines),
        "compiler_all_reduce_count": _operation_count(compiler_hlo, "all-reduce"),
        "compiler_all_gather_count": len(all_gather_lines),
        "sparse_core_reduce_scatter_count": sum(
            "reduce_scatter_offload_config" in line
            and '"device_type":"DEVICE_TYPE_SPARSECORE"' in line
            for line in reduce_scatter_lines
        ),
        "sparse_core_all_gather_count": sum(
            "all_gather_offload_config" in line and '"device_type":"DEVICE_TYPE_SPARSECORE"' in line
            for line in all_gather_lines
        ),
    }


def _validate_analysis(
    analysis: dict[str, Any],
    stablehlo_path: Path,
    compiler_hlo_path: Path,
    strategy: str,
) -> None:
    _expect_keys(
        analysis,
        {
            "analysis_schema",
            "stablehlo_sha256",
            "compiler_hlo_sha256",
            "cost_metrics",
            "memory",
            "collectives",
        },
        "COMPILER_ANALYSIS",
    )
    if (
        analysis["analysis_schema"] != _COMPILER_ANALYSIS_SCHEMA
        or analysis["stablehlo_sha256"] != _file_sha256(stablehlo_path)
        or analysis["compiler_hlo_sha256"] != _file_sha256(compiler_hlo_path)
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_COMPILER_ANALYSIS_HASH_MISMATCH")
    metrics = analysis["cost_metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_COMPILER_METRICS_INVALID")
    names: list[str] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            raise TypeError("SURFACE_COMPILE_INDEPENDENT_COMPILER_METRIC_INVALID")
        _expect_keys(metric, {"name", "raw_value", "value", "available"}, "COMPILER_METRIC")
        name = metric["name"]
        raw = metric["raw_value"]
        value = metric["value"]
        available = metric["available"]
        if (
            not isinstance(name, str)
            or not name
            or isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(raw)
            or not isinstance(available, bool)
            or (available and value != raw)
            or (not available and (name != "optimal_seconds" or raw >= 0 or value is not None))
        ):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_COMPILER_METRIC_INVALID")
        names.append(name)
    if names != sorted(set(names)):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_COMPILER_METRIC_ORDER_MISMATCH")
    memory = analysis["memory"]
    if not isinstance(memory, dict):
        raise TypeError("SURFACE_COMPILE_INDEPENDENT_COMPILER_MEMORY_INVALID")
    _expect_keys(memory, _ANALYSIS_MEMORY_KEYS, "COMPILER_MEMORY")
    for key in _ANALYSIS_MEMORY_KEYS - {"buffer_assignment_available", "buffer_assignment_sha256"}:
        _require_int(memory[key], f"COMPILER_MEMORY_{key.upper()}")
    available = memory["buffer_assignment_available"]
    digest = memory["buffer_assignment_sha256"]
    if (
        not isinstance(available, bool)
        or (
            available
            and (
                memory["buffer_assignment_size_bytes"] == 0
                or not isinstance(digest, str)
                or _HEX_64.fullmatch(digest) is None
            )
        )
        or (not available and (memory["buffer_assignment_size_bytes"] != 0 or digest is not None))
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_COMPILER_BUFFER_ASSIGNMENT_INVALID")
    collectives = analysis["collectives"]
    if not isinstance(collectives, dict):
        raise TypeError("SURFACE_COMPILE_INDEPENDENT_COMPILER_COLLECTIVES_INVALID")
    _expect_keys(collectives, _COLLECTIVE_KEYS, "COMPILER_COLLECTIVES")
    for key in _COLLECTIVE_KEYS:
        _require_int(collectives[key], f"COMPILER_COLLECTIVE_{key.upper()}")
    stablehlo = stablehlo_path.read_text()
    compiler_hlo = compiler_hlo_path.read_text()
    if collectives != _collectives(stablehlo, compiler_hlo):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_COMPILER_COLLECTIVES_MISMATCH")
    expected = (
        {
            "stablehlo_reduce_scatter_count": 1,
            "stablehlo_all_gather_count": 0,
            "compiler_reduce_scatter_count": 1,
            "compiler_all_reduce_count": 0,
            "compiler_all_gather_count": 0,
            "sparse_core_reduce_scatter_count": 1,
            "sparse_core_all_gather_count": 0,
        }
        if strategy == "xla_reduce_scatter"
        else {key: 0 for key in _COLLECTIVE_KEYS}
    )
    if collectives != expected:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_COMPILER_STRATEGY_MISMATCH")


def _validate_capture(
    root: Path,
    contract: dict[str, Any],
    result: dict[str, Any],
    envelope: dict[str, Any],
    scenario: dict[str, Any],
    strategy: str,
) -> tuple[dict[str, Any], VerifiedCompileCapture]:
    _expect_keys(
        envelope,
        {
            "capture",
            "abstract_input_abi",
            "stablehlo_path",
            "compiler_hlo_path",
            "compiler_analysis_path",
            "compiler_analysis",
        },
        "CAPTURE_ENVELOPE",
    )
    capture = envelope["capture"]
    abi = envelope["abstract_input_abi"]
    analysis = envelope["compiler_analysis"]
    if not isinstance(capture, dict) or not isinstance(abi, dict) or not isinstance(analysis, dict):
        raise TypeError("SURFACE_COMPILE_INDEPENDENT_CAPTURE_INVALID")
    _expect_keys(capture, _CAPTURE_KEYS, "CAPTURE")
    _expect_keys(abi, _ABI_KEYS, "ABSTRACT_INPUT_ABI")
    repetition = result["repetition"]
    base = f"repetition-{repetition}/arms/{scenario['name']}/{strategy}"
    stable_relative = f"{base}/stablehlo.txt"
    compiler_relative = f"{base}/compiler_hlo.txt"
    analysis_relative = f"{base}/compiler_analysis.json"
    if (
        envelope["stablehlo_path"] != stable_relative
        or envelope["compiler_hlo_path"] != compiler_relative
        or envelope["compiler_analysis_path"] != analysis_relative
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_CAPTURE_PATH_MISMATCH")
    expected_abi = {
        "lhs_shape": [scenario["m"], scenario["k"]],
        "lhs_dtype": "bfloat16",
        "lhs_sharding": "PartitionSpec(None, 't')",
        "rhs_shape": [scenario["k"], scenario["n"]],
        "rhs_dtype": "bfloat16",
        "rhs_sharding": "PartitionSpec('t', None)",
        "output_shape": [scenario["m"], scenario["n"]],
        "output_dtype": "float32",
        "output_sharding": "PartitionSpec(None, 't')",
        "schema_version": contract["compile_input_abi_schema"],
    }
    if abi != expected_abi:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_ABSTRACT_INPUT_ABI_MISMATCH")
    workload = _semantic_sha256(
        "matmul-collective-surface-input",
        _identity_sha256(contract),
        scenario["name"],
        str(scenario["m"]),
        str(scenario["k"]),
        str(scenario["n"]),
    )
    lhs = _semantic_sha256(workload, "lhs", contract["input_dtype"])
    rhs = _semantic_sha256(workload, "rhs", contract["input_dtype"])
    expected_input = _semantic_sha256(lhs, rhs)
    for key in (
        "input_contract_sha256",
        "distributed_schedule_sha256",
        "physical_schedule_sha256",
        "pallas_source_sha256",
        "stablehlo_sha256",
        "semantic_stablehlo_sha256",
        "compiler_hlo_sha256",
        "semantic_compiler_hlo_sha256",
    ):
        _require_hex(capture[key], _HEX_64, f"CAPTURE_{key.upper()}")
    if (
        capture["scenario_name"] != scenario["name"]
        or capture["strategy"] != strategy
        or capture["repetition"] != repetition
        or capture["status"] != "succeeded"
        or capture["error_sha256"] is not None
        or capture["input_contract_sha256"] != expected_input
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_CAPTURE_IDENTITY_MISMATCH")
    stable_path = root / stable_relative
    compiler_path = root / compiler_relative
    analysis_path = root / analysis_relative
    stablehlo = stable_path.read_text()
    compiler_hlo = compiler_path.read_text()
    if capture["stablehlo"] != stablehlo or capture["compiler_hlo"] != compiler_hlo:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_RAW_HLO_MISMATCH")
    if (
        capture["stablehlo_sha256"] != _sha256_bytes(stablehlo.encode())
        or capture["semantic_stablehlo_sha256"]
        != _sha256_bytes((stablehlo.rstrip("\n") + "\n").encode())
        or capture["compiler_hlo_sha256"] != _sha256_bytes(compiler_hlo.encode())
        or capture["semantic_compiler_hlo_sha256"]
        != _sha256_bytes(_semantic_compiler_hlo(compiler_hlo).encode())
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_HLO_HASH_MISMATCH")
    _validate_static_abi(stablehlo, compiler_hlo, scenario)
    recorded_analysis = _read_json(analysis_path)
    if analysis != recorded_analysis:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_ANALYSIS_ENVELOPE_MISMATCH")
    _validate_analysis(recorded_analysis, stable_path, compiler_path, strategy)
    return capture, VerifiedCompileCapture(
        repetition=repetition,
        scenario_name=scenario["name"],
        strategy=strategy,
        abstract_input_abi_sha256=_identity_sha256(abi),
        stablehlo_path=stable_relative,
        stablehlo_sha256=_file_sha256(stable_path),
        compiler_hlo_path=compiler_relative,
        compiler_hlo_sha256=_file_sha256(compiler_path),
        compiler_analysis_path=analysis_relative,
        compiler_analysis_sha256=_file_sha256(analysis_path),
    )


def _validate_workers(
    root: Path,
    contract: dict[str, Any],
    identity: dict[str, Any],
    execution_authority_sha256: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[VerifiedCompileCapture, ...]]:
    captures: list[dict[str, Any]] = []
    verified: list[VerifiedCompileCapture] = []
    pids: set[int] = set()
    semantic_by_arm: dict[tuple[str, str], set[tuple[str, str]]] = {}
    scenarios = contract["scenarios"]
    strategies = contract["strategies"]
    for repetition in (1, 2):
        request = _read_json(root / f"repetition-{repetition}/request.json")
        _expect_keys(
            request,
            {
                "attempt_id",
                "repetition",
                "invocation_nonce",
                "authority_sha256",
                "compilation_cache_schema",
                "contract",
            },
            "WORKER_REQUEST",
        )
        nonce = _require_hex(request["invocation_nonce"], _HEX_64, "WORKER_NONCE")
        if (
            request["attempt_id"] != identity["attempt_id"]
            or request["repetition"] != repetition
            or request["authority_sha256"] != execution_authority_sha256
            or request["compilation_cache_schema"] != "isolated-empty-temporary-directory-v1"
            or request["contract"] != contract
        ):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_WORKER_REQUEST_MISMATCH")
        started = _read_json(root / f"repetition-{repetition}/STARTED.json")
        if started != {
            "attempt_id": identity["attempt_id"],
            "invocation_nonce": nonce,
            "repetition": repetition,
            "state": "started",
        }:
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_WORKER_START_MISMATCH")
        result = _read_json(root / f"repetition-{repetition}/result.json")
        _expect_keys(
            result,
            {
                "attempt_id",
                "repetition",
                "invocation_nonce",
                "worker_pid",
                "authority_sha256",
                "captures",
            },
            "WORKER_RESULT",
        )
        pid = _require_int(result["worker_pid"], "WORKER_PID", minimum=1)
        envelopes = result["captures"]
        if (
            pid in pids
            or result["attempt_id"] != identity["attempt_id"]
            or result["repetition"] != repetition
            or result["invocation_nonce"] != nonce
            or result["authority_sha256"] != execution_authority_sha256
            or not isinstance(envelopes, list)
            or len(envelopes) != len(scenarios) * len(strategies)
        ):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_WORKER_RESULT_MISMATCH")
        pids.add(pid)
        for envelope, (scenario, strategy) in zip(
            envelopes,
            ((scenario, strategy) for scenario in scenarios for strategy in strategies),
            strict=True,
        ):
            if not isinstance(envelope, dict):
                raise TypeError("SURFACE_COMPILE_INDEPENDENT_CAPTURE_ENVELOPE_INVALID")
            capture, verified_capture = _validate_capture(
                root, contract, result, envelope, scenario, strategy
            )
            captures.append(capture)
            verified.append(verified_capture)
            semantic_by_arm.setdefault((scenario["name"], strategy), set()).add(
                (
                    capture["semantic_stablehlo_sha256"],
                    capture["semantic_compiler_hlo_sha256"],
                )
            )
    if any(len(values) != 1 for values in semantic_by_arm.values()):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_UNSTABLE_COMPILER_HASH")
    identity_by_arm: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for capture in captures:
        identity_by_arm.setdefault((capture["scenario_name"], capture["strategy"]), set()).add(
            (
                capture["distributed_schedule_sha256"],
                capture["physical_schedule_sha256"],
                capture["pallas_source_sha256"],
            )
        )
    if any(len(values) != 1 for values in identity_by_arm.values()):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_UNSTABLE_ARM_IDENTITY")
    observed_identities = {key: next(iter(values)) for key, values in identity_by_arm.items()}
    if observed_identities != _EXPECTED_ARM_IDENTITIES:
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_CANONICAL_ARM_IDENTITY_MISMATCH")
    return tuple(captures), tuple(verified)


def _canonical_report_captures(
    captures: tuple[dict[str, Any], ...],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    by_key = {
        (capture["scenario_name"], capture["strategy"], capture["repetition"]): capture
        for capture in captures
    }
    expected_keys = tuple(
        (scenario["name"], strategy, repetition)
        for scenario in contract["scenarios"]
        for strategy in contract["strategies"]
        for repetition in (1, 2)
    )
    if set(by_key) != set(expected_keys):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_REPORT_CAPTURE_INVENTORY_MISMATCH")
    return tuple(by_key[key] for key in expected_keys)


def _validate_report(
    root: Path,
    contract: dict[str, Any],
    design_id: str,
    source_authority_sha256: str,
    execution_authority_sha256: str,
    worker_captures: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], str]:
    report = _read_json(root / "compile_report.json")
    _expect_keys(
        report,
        {
            "schema_version",
            "design_id",
            "source_authority_sha256",
            "execution_authority_sha256",
            "captures",
        },
        "COMPILE_REPORT",
    )
    expected_captures = list(_canonical_report_captures(worker_captures, contract))
    if (
        report["schema_version"] != _REPORT_SCHEMA
        or report["design_id"] != design_id
        or report["source_authority_sha256"] != source_authority_sha256
        or report["execution_authority_sha256"] != execution_authority_sha256
        or report["captures"] != expected_captures
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_REPORT_REPLAY_MISMATCH")
    return report, _identity_sha256(report)


def _validate_ledger(
    path: Path,
    identity: dict[str, Any],
    design_id: str,
    authority: dict[str, Any],
    source_authority_sha256: str,
    execution_authority_sha256: str,
    report: dict[str, Any],
    report_sha256: str,
) -> tuple[str, ...]:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_LEDGER_INTEGRITY_MISMATCH")
        objects = tuple(
            connection.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_autoindex_%' ORDER BY type, name"
            )
        )
        if objects != (("table", "events"), ("table", "sqlite_sequence")):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_LEDGER_SCHEMA_MISMATCH")
        columns = tuple(
            (row[1], row[2], row[3], row[5])
            for row in connection.execute("PRAGMA table_info(events)")
        )
        if columns != (
            ("sequence", "INTEGER", 0, 1),
            ("run_id", "TEXT", 1, 0),
            ("state", "TEXT", 1, 0),
            ("timestamp_ns", "INTEGER", 1, 0),
            ("payload_sha256", "TEXT", 1, 0),
        ):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_LEDGER_COLUMNS_MISMATCH")
        indexes = tuple(
            (row[1], row[2], row[3], row[4])
            for row in connection.execute("PRAGMA index_list(events)")
        )
        if indexes != (("sqlite_autoindex_events_1", 1, "u", 0),):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_LEDGER_INDEX_MISMATCH")
        rows = tuple(
            connection.execute(
                "SELECT sequence, run_id, state, timestamp_ns, payload_sha256 "
                "FROM events ORDER BY sequence"
            )
        )
        if tuple(connection.execute("SELECT name, seq FROM sqlite_sequence")) != (("events", 4),):
            raise ValueError("SURFACE_COMPILE_INDEPENDENT_LEDGER_SEQUENCE_MISMATCH")
    if (
        len(rows) != 4
        or tuple(row[0] for row in rows) != (1, 2, 3, 4)
        or any(row[1] != identity["attempt_id"] for row in rows)
        or tuple(row[2] for row in rows) != _FINAL_STATES
        or any(
            isinstance(row[3], bool) or not isinstance(row[3], int) or row[3] < 0 for row in rows
        )
        or tuple(row[3] for row in rows) != tuple(sorted(row[3] for row in rows))
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_LEDGER_STATE_MISMATCH")
    repetition_one = {
        f"{capture['scenario_name']}:{capture['strategy']}": [
            capture["physical_schedule_sha256"],
            capture["pallas_source_sha256"],
        ]
        for capture in report["captures"]
        if capture["repetition"] == 1
    }
    expected_payloads = (
        {
            "design_id": design_id,
            "execution_authority_sha256": execution_authority_sha256,
            "attempt_claim_path": identity["attempt_claim_path"],
            "attempt_claim_sha256": identity["attempt_claim_sha256"],
        },
        {
            "source_authority_sha256": source_authority_sha256,
            "executor_source_sha256": authority["executor_source_sha256"],
            "worker_source_sha256": authority["worker_source_sha256"],
            "verifier_source_sha256": authority["verifier_source_sha256"],
            "devices": authority["devices"],
        },
        {"arm_identities_sha256": _identity_sha256(repetition_one)},
        {"compile_report_sha256": report_sha256},
    )
    if tuple(row[4] for row in rows) != tuple(
        _identity_sha256(payload) for payload in expected_payloads
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_LEDGER_PAYLOAD_MISMATCH")
    return _FINAL_STATES


def verify_surface_compile_independently(
    root: Path,
    contract_path: Path,
) -> SurfaceCompileVerification:
    if root.is_symlink():
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_ROOT_INVALID")
    root = root.resolve(strict=True)
    _validate_archive_tree(root)
    if (root / "failure.json").exists():
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_ATTEMPT_INCOMPLETE")
    manifest, _ = _validate_manifest(root)
    contract, design_id = _validate_contract(contract_path, root)
    authority, source_authority_sha256, execution_authority_sha256 = (
        _validate_source_and_execution_authority(root, contract)
    )
    identity = _validate_run_identity(
        root,
        contract,
        design_id,
        execution_authority_sha256,
        authority["source"]["source_commit"],
    )
    worker_captures, verified_captures = _validate_workers(
        root, contract, identity, execution_authority_sha256
    )
    report, report_sha256 = _validate_report(
        root,
        contract,
        design_id,
        source_authority_sha256,
        execution_authority_sha256,
        worker_captures,
    )
    ledger_path = root / "ledger.sqlite"
    ledger_sha256 = _file_sha256(ledger_path)
    ledger_states = _validate_ledger(
        ledger_path,
        identity,
        design_id,
        authority,
        source_authority_sha256,
        execution_authority_sha256,
        report,
        report_sha256,
    )
    manifest_identity = manifest["identity"]
    if (
        manifest_identity != identity
        or manifest["report_sha256"] != report_sha256
        or manifest["ledger_sha256"] != ledger_sha256
    ):
        raise ValueError("SURFACE_COMPILE_INDEPENDENT_MANIFEST_AUTHORITY_MISMATCH")
    verifier_source_sha256 = _file_sha256(Path(__file__))
    return SurfaceCompileVerification(
        attempt_id=identity["attempt_id"],
        design_id=design_id,
        source_authority_sha256=source_authority_sha256,
        execution_authority_sha256=execution_authority_sha256,
        compile_report_sha256=report_sha256,
        ledger_sha256=ledger_sha256,
        ledger_states=ledger_states,
        verifier_source_sha256=verifier_source_sha256,
        verifier_canonically_bound=True,
        captures=verified_captures,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    print(verify_surface_compile_independently(args.root, args.contract).as_json())


if __name__ == "__main__":
    main()
