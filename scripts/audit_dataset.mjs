import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const root = path.resolve(process.argv[2] || 'datasets');
const out = path.resolve(process.argv[3] || 'data/reports');
const processed = path.resolve('data/processed');
fs.mkdirSync(out, { recursive: true }); fs.mkdirSync(processed, { recursive: true });

function parseCSV(text) {
  const rows=[]; let row=[], field='', quoted=false;
  for (let i=0;i<text.length;i++) {
    const c=text[i];
    if (quoted) { if(c==='"' && text[i+1]==='"'){field+='"';i++;} else if(c==='"') quoted=false; else field+=c; }
    else if(c==='"' && field==='') quoted=true;
    else if(c===','){row.push(field);field='';}
    else if(c==='\n'){row.push(field.replace(/\r$/,'')); rows.push(row);row=[];field='';}
    else field+=c;
  }
  if(field.length || row.length){row.push(field.replace(/\r$/,''));rows.push(row);}
  if(!rows.length)return {headers:[],records:[]};
  const headers=rows.shift().map((x)=>x.replace(/^\uFEFF/, '').trim());
  return {headers,records:rows.filter(r=>r.some(v=>v.trim())).map(values=>Object.fromEntries(headers.map((h,i)=>[h, values[i]??''])))};
}
const files=fs.readdirSync(root).filter(f=>f.toLowerCase().endsWith('.csv')).sort();
const report={created_at:new Date().toISOString(),dataset_directory:root,file_count:files.length,total_rows:0,files:[],duplicate_source_urls:[],duplicate_content_groups:[],overlaps:[],questionable_records:[]};
const urlMap=new Map(), contentMap=new Map(), associationMap=new Map();
for(const file of files){
  const raw=fs.readFileSync(path.join(root,file)); const decoded=raw.toString('utf8'); const {headers,records}=parseCSV(decoded);
  const missing=Object.fromEntries(headers.map(h=>[h,0])); const typeCounts=Object.fromEntries(headers.map(h=>[h,{}]));
  let generic=0,missingTitle=0,missingText=0,unverified=0,invalidUrls=0,malformed=0,duplicateRows=0;
  const seenRows=new Set(); const statuses={};
  for(let i=0;i<records.length;i++){
    const r=records[i];
    const rowHash=crypto.createHash('sha256').update(JSON.stringify(Object.keys(r).sort().map(k=>[k,String(r[k]).replace(/\s+/g,' ').trim()]))).digest('hex'); if(seenRows.has(rowHash))duplicateRows++;else seenRows.add(rowHash);
    for(const h of headers){const v=r[h]??'';if(!String(v).trim())missing[h]++; const t=!String(v).trim()?'empty':/^\d{4}-\d{2}-\d{2}$/.test(v)?'date':/^-?\d+(\.\d+)?$/.test(v)?'numeric':'string';typeCounts[h][t]=(typeCounts[h][t]||0)+1;}
    const title=String(r.case_title??r['Case Title']??r.page_title??'').trim();
    if(!title)missingTitle++; else if(/^(full document|document|unknown|n\/a|case)$/i.test(title))generic++;
    const body=String(r.full_text??r['Relevant Judgment Text']??r.case_summary??r['Brief Judgment Summary']??r.description??'').trim();if(body.length<80)missingText++;
    const status=String(r.verification_status??r['Verification Status']??'').trim(); if(status){statuses[status]=(statuses[status]||0)+1;unverified++;}
    const url=String(r.source_url??r.case_url??r['Original Judgment URL']??'').trim();
    if(url){let u;try{u=new URL(url);if(!['http:','https:'].includes(u.protocol))throw 0;}catch{invalidUrls++;}
      const k=url.toLowerCase().replace(/\/$/,''); if(!urlMap.has(k))urlMap.set(k,[]);urlMap.get(k).push({file,row:i+2,title:title.slice(0,120)});
    }
    if(body.length>100){const hash=crypto.createHash('sha256').update(body.replace(/\s+/g,' ').toLowerCase()).digest('hex');if(!contentMap.has(hash))contentMap.set(hash,[]);contentMap.get(hash).push({file,row:i+2,title:title.slice(0,120)});}
    const provision=String(r.provision_id??r['Provision ID']??r.legal_provision??r.provision??'').trim();
    if(url && provision){const key=url.toLowerCase().replace(/\/$/,'');if(!associationMap.has(key))associationMap.set(key,new Set());associationMap.get(key).add(provision);}
    const suspicious=(status && !/(verified|official|primary)/.test(status)) || !title || title==='Full Document' || body.length<80 || invalidUrls>0;
    if(suspicious && report.questionable_records.length<500)report.questionable_records.push({file,row:i+2,title:title||null,verification_status:status||null,source_url:url||null,text_length:body.length,reason:!title?'missing title':title==='Full Document'?'generic title':body.length<80?'missing/short text':(status&&!/(verified|official|primary)/.test(status))?'verification requires review':'check metadata'});
    malformed += headers.some(h=>!(h in r))?1:0;
  }
  report.total_rows+=records.length; report.files.push({name:file,bytes:raw.length,rows:records.length,columns:headers,types:typeCounts,missing_values:missing,duplicate_rows:duplicateRows,generic_titles:generic,missing_titles:missingTitle,missing_document_text:missingText,records_with_source_verification_label_needing_independent_check:unverified,verification_status_values:statuses,invalid_urls:invalidUrls,malformed_rows:malformed});
}
for(const [url,items] of urlMap)if(items.length>1)report.duplicate_source_urls.push({url,count:items.length,records:items});
for(const [hash,items] of contentMap)if(items.length>1)report.duplicate_content_groups.push({sha256:hash,count:items.length,records:items});
for(const [url,provisions] of associationMap)if(provisions.size>1)report.overlaps.push({url,provisions:[...provisions]});
report.duplicate_source_url_group_count=report.duplicate_source_urls.length;report.duplicate_content_group_count=report.duplicate_content_groups.length;report.overlap_url_count=report.overlaps.length;
report.notes=['The input dataset contains candidate records; labels and URL presence do not establish legal authority.','CSV inference reports observed lexical value types; it does not establish legal field validity.','All supplied verification labels are reported verbatim and need independent source checks. No record was promoted to verified.','Provision-to-document relevance requires substantive legal review; URLs that recur across multiple provision labels are reported as overlap candidates, not automatically classified as errors.'];
fs.writeFileSync(path.join(out,'dataset_audit.json'),JSON.stringify(report,null,2));
const lines=['# Dataset audit','',`Created: ${report.created_at}`,`Files: ${report.file_count}`,`Rows: ${report.total_rows}`,`Duplicate URL groups: ${report.duplicate_source_url_group_count}`,`Duplicate content groups: ${report.duplicate_content_group_count}`,`URLs across multiple provision labels: ${report.overlap_url_count}`,'','## Per file','',...report.files.map(f=>`- ${f.name}: ${f.rows} rows; ${f.columns.length} columns; ${f.duplicate_rows} exact duplicate rows; ${f.generic_titles} generic titles; ${f.missing_document_text} short/missing text rows; ${f.records_with_source_verification_label_needing_independent_check} records with supplied status labels; ${f.invalid_urls} invalid URLs`),'','## Caveats','',...report.notes.map(x=>`- ${x}`)];
fs.writeFileSync(path.join(out,'dataset_audit.md'),lines.join('\n'));
console.log(JSON.stringify({files:report.file_count,rows:report.total_rows,duplicate_url_groups:report.duplicate_source_url_group_count,duplicate_content_groups:report.duplicate_content_group_count,overlapping_urls:report.overlap_url_count,report:path.join(out,'dataset_audit.md')}));
