/**
 * Image handling for the composer.
 *
 * Attachments stay local to the browser — nothing is uploaded, and nothing is
 * sent anywhere until the claim is checked. Large photographs are scaled down
 * before they are held in state, because a 12-megapixel phone photo becomes a
 * ~7 MB string as a data URL and would be carried around by every render.
 */

export const MAX_IMAGE_BYTES = 8 * 1024 * 1024
const MAX_EDGE = 1600
const SHRINK_ABOVE_BYTES = 1.5 * 1024 * 1024

export class ImageError extends Error {
  constructor(userMessage) {
    super(userMessage)
    this.name = 'ImageError'
    this.userMessage = userMessage
  }
}

function readAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new ImageError('That file could not be read.'))
    reader.readAsDataURL(file)
  })
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new ImageError('That image could not be opened.'))
    image.src = dataUrl
  })
}

function downscale(image) {
  const scale = MAX_EDGE / Math.max(image.width, image.height)
  const width = Math.round(image.width * scale)
  const height = Math.round(image.height * scale)

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  context.imageSmoothingQuality = 'high'
  context.drawImage(image, 0, 0, width, height)

  return { dataUrl: canvas.toDataURL('image/jpeg', 0.85), width, height }
}

/** Reads a File into `{ dataUrl, name, width, height }`, scaled if needed. */
export async function readImageFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    throw new ImageError('That file isn’t an image. Attach a PNG, JPEG or WebP.')
  }
  if (file.size > MAX_IMAGE_BYTES) {
    throw new ImageError('That image is larger than 8 MB. Try a smaller version.')
  }

  const original = await readAsDataURL(file)
  const image = await loadImage(original)

  const isHuge = Math.max(image.width, image.height) > MAX_EDGE
  const isHeavy = file.size > SHRINK_ABOVE_BYTES

  if (!isHuge && !isHeavy) {
    return { dataUrl: original, name: file.name, width: image.width, height: image.height }
  }

  const scaled = downscale(image)
  return { ...scaled, name: file.name }
}

/** The first image in a drop or paste payload, if there is one. */
export function firstImageFile(list) {
  const files = Array.from(list ?? [])
  return files.find((file) => file.type?.startsWith('image/')) ?? null
}
