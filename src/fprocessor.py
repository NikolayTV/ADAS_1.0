import queue
import os, fnmatch
import asyncio


import cv2
import sys
import time 
import base64
import requests

class aresult_output:
    def write_record(self):
        pass


class vaprocessing():
    analytic_result = []
    ao = aresult_output()

    def __init__(self,filename):
        self.filename = filename
        self.exit_flag = False
        self.finished = False

    def main(self):
        cap = cv2.VideoCapture(self.filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        url = 'http://0.0.0.0:7777/'
        # prepare headers for http request
        content_type = 'image/jpeg'
        headers = {'content-type': content_type}

        wait_ms = 20
        while not self.exit_flag:
            ret, frame = cap.read()
            if ret:
                frame_start_time = time.time()
                retval, buffer = cv2.imencode('.jpg', frame)
                frame_encoded = base64.b64encode(buffer)
                response = requests.post(url, data={"image":frame_encoded})
                rr  = response.json()
                print(rr)
                self.analytic_result.append(rr)       

                if self.ao != None:
                    self.ao.write_record(rr)        
                # if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                #     break

                now = time.time()
                frame_time = now - frame_start_time
                fps = 1.0 / frame_time
                print(fps)
            else:
                print("Error read frame")
                break

        self.finished = True

class Output(aresult_output):
    def __init__(self,name):
        self.fname = f'{name}.json'

    def report(fname,data):
        try:
            with open(fname, "r+", newline='') as f:
                f.seek(0,2)
                # for line in data:
                    # print(line)
                    # writer.writerow(line)
                f.write(data)
        except:
            try:
                with open(fname, "w", newline='') as f:
                    f.write(data)
            except:
                print("Output error")
                pass

    def write_record(self,rec):
        Output.report(self.fname,rec)
        # with open(f, "at", newline='',encoding='utf-8') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerow(hdr)
        #     print("Header done")

        # f = open(self.fname,'+aw')
        # f.write(vap.analytic_result)
        # f.close()


class FilePool:
    files = dict()
    _fileq = asyncio.Queue()
    # _event = asyncio.Event()
    _tasks = []

    def __init__(self):
        self.loop = asyncio.get_event_loop()

    async def get_files(self):
        return self.files

    async def add_dir(self,path):
        list = os.listdir(path)
        # for f in enumerate(list):
        #     name = (f[1])
        #     await self._fileq.put(name) 

        pattern = "*.mp4"  
        for entry in list:  
            if fnmatch.fnmatch(entry, pattern):
                await self._fileq.put(f'{path}{entry}') 

        await self._fileq.put({'cmd':'done'}) 
        

    def add_file(self,desc):
        if 'filename' in desc:
            fname = desc["filename"]
        self._fileq.put(desc) 


    def create_stream(self,file):
        return False

    async def cleanup_task(self):
        pass

    def process_chunk(self,fid):
        print(f'File process {fid}')

        o = Output(fid)
        vap = vaprocessing(fid)
        vap.ao = o
        vap.main()
        # foutput = f'{fid}.json'
        
        # f = open(foutput,'w')
        # f.write(vap.analytic_result)
        # f.close()
        return True


    async def worker(self, config):
        flag = True
        print("worker.starts()")
        while flag:
            while not self._fileq.empty():
                f = await self._fileq.get()
                # print(f'file in queue:{f}')
                if f == {'cmd':'done'}:
                    flag = False
                    break
                # print(f'{f}')
                self.process_chunk(f)
                self._fileq.task_done()
        print("worker.ends()")
        # self.proc_cfg.terminate()

    async def _start(self):
        print("_start")
        # self.loop.run_until_complete(
        #     self.worker({})
        # )
        # await self.worker({})
        task = asyncio.create_task(self.worker({}))
        self._tasks.append(task)

    def start(self):
        print("start")

        asyncio.run(self._start())
        # asyncio.ensure_future(self._start())
        return True

    async def _stop(self):
        print("_stop")
        await self._fileq.put({'cmd':'done'}) 
        for task in self._tasks:
            task.cancel()
        # asyncio.gather(*self._tasks, return_exceptions=True)

    def stop(self):
        print("stop")
        # self._event.set()
        asyncio.run(self._stop())
        return True

    # def do(self):
    #     loop = asyncio.get_event_loop()
    #     loop.run_until_complete(self.worker({}))
    #     loop.close()
        


if __name__ == '__main__':
    pool =  FilePool()
    asyncio.run(pool.add_dir('/opt/video/'))
    pool.start()
    print("--------------------------")
    pool.stop()
