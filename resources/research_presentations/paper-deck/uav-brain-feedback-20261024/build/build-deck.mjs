import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const base=path.resolve(process.argv[2]);
const skill='/Users/xiangjutao/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const py='/Users/xiangjutao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const names=['01-experiment','02-architecture','03-gap','04-method'];
const noteText=await fs.readFile(path.join(base,'speaker-notes.md'),'utf8');
const notes=noteText.split(/\n## /).slice(1);
for(let i=0;i<names.length;i++){
 const bytes=await fs.readFile(path.join(base,'images',names[i]+'.png'));
 const slide=presentation.slides.add();
 slide.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:names[i],fit:'contain',position:{left:0,top:0,width:1280,height:720}});
 slide.speakerNotes.text=notes[i]||noteText;
 const png=await presentation.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(base,'build','render-'+(i+1)+'.png'),new Uint8Array(await png.arrayBuffer()));
}
const candidatePath=path.join(base,'build','candidate.pptx');
await(await PresentationFile.exportPptx(presentation)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools','artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir:base,candidatePath,finalPath:path.join(base,'output','无人机大小脑协同_研究汇报.pptx'),pythonExecutable:py,integrityValidatorPath:path.join(skill,'container_tools','inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools','inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000'],explicitTotalSlideCount:4,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],verifyArtifactToolImport:true,receiptPath:path.join(base,'build','validation.json')});
console.log(JSON.stringify(result));
