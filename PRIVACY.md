# Privacidad

_Última actualización: 2026-08-21_

## Lo corto

**Esku funciona entero en tu dispositivo. El vídeo de tu cámara no se graba, no se sube y no
sale del navegador.** No hay servidor al que llegue, porque no hay servidor.

No usamos cookies, ni analítica, ni publicidad, ni identificadores. Por eso esta app no te pide
aceptar nada al entrar: no hay nada que aceptar.

## Qué pasa con la cámara

Los fotogramas se analizan en el momento, dentro del navegador, para localizar tus manos, tu
postura y tu cara. De ahí sale una lista de coordenadas que alimenta al modelo, y tanto los
fotogramas como las coordenadas se descartan al instante. **Nada de eso se guarda ni se envía.**

El permiso de cámara lo concedes al navegador, no a nosotros, y puedes retirarlo cuando quieras
desde la configuración del navegador.

## Qué se guarda en tu dispositivo

Solo si tú lo creas:

- **Los signos que enseñas.** Cuando grabas un signo propio, se guardan las coordenadas de esas
  grabaciones en el almacenamiento local del navegador (IndexedDB). No hay vídeo, solo números.
- **Los modelos descargados**, para que la app arranque rápido y funcione sin conexión.

Todo eso vive en tu navegador. Puedes **borrar cualquier signo enseñado desde la propia app**, y
borrarlo todo vaciando los datos del sitio en la configuración del navegador.

## Lo único que sale de tu dispositivo

Esku se sirve como página estática desde **GitHub Pages**. Como cualquier web, al descargarla tu
navegador se conecta a los servidores de GitHub, que registran la petición y tu dirección IP en
sus propios registros. Eso ocurre antes de que la app haga nada, no lo controlamos y no lo
recibimos. GitHub es una empresa estadounidense; su política de privacidad es la que aplica a
esos registros.

Después de esa descarga, la app no vuelve a hablar con ningún servidor.

## Tus derechos

El RGPD te da derecho a acceder a tus datos, corregirlos y borrarlos. Como no recibimos ninguno,
aquí el ejercicio de esos derechos es directo y no depende de nosotros: los signos que enseñas
los borras tú desde la app o vaciando los datos del sitio.

Si algo de esto no te cuadra o crees que la app hace algo distinto de lo que dice, ábrenos una
incidencia: <https://github.com/Endika/esku/issues>. El código es público y se puede comprobar.

## Una nota sobre por qué esto está escrito así

Esku está pensada para que una persona sorda pueda hacerse entender en una consulta médica. Lo
que se signa ahí es información de salud, que es de lo más sensible que hay. La respuesta a eso
no es prometer que lo guardamos bien: es **no tenerlo**. Por eso todo el reconocimiento corre en
el dispositivo, aunque salga más lento que en un servidor.
