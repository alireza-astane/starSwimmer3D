#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.18059834733211774,-0.09177549654042169,-0.5969504716961467>, 1 }        
    sphere {  m*<0.23370676792833345,0.12973454800791803,4.5446312894023215>, 1 }
    sphere {  m*<2.5541100466741393,0.010258478845952476,-1.8261599971473292>, 1 }
    sphere {  m*<-1.8022137072250077,2.2366984478781777,-1.5708962371121158>, 1}
    sphere { m*<-1.534426486187176,-2.6509934945257196,-1.3813499519495431>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23370676792833345,0.12973454800791803,4.5446312894023215>, <-0.18059834733211774,-0.09177549654042169,-0.5969504716961467>, 0.5 }
    cylinder { m*<2.5541100466741393,0.010258478845952476,-1.8261599971473292>, <-0.18059834733211774,-0.09177549654042169,-0.5969504716961467>, 0.5}
    cylinder { m*<-1.8022137072250077,2.2366984478781777,-1.5708962371121158>, <-0.18059834733211774,-0.09177549654042169,-0.5969504716961467>, 0.5 }
    cylinder {  m*<-1.534426486187176,-2.6509934945257196,-1.3813499519495431>, <-0.18059834733211774,-0.09177549654042169,-0.5969504716961467>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.18059834733211774,-0.09177549654042169,-0.5969504716961467>, 1 }        
    sphere {  m*<0.23370676792833345,0.12973454800791803,4.5446312894023215>, 1 }
    sphere {  m*<2.5541100466741393,0.010258478845952476,-1.8261599971473292>, 1 }
    sphere {  m*<-1.8022137072250077,2.2366984478781777,-1.5708962371121158>, 1}
    sphere { m*<-1.534426486187176,-2.6509934945257196,-1.3813499519495431>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23370676792833345,0.12973454800791803,4.5446312894023215>, <-0.18059834733211774,-0.09177549654042169,-0.5969504716961467>, 0.5 }
    cylinder { m*<2.5541100466741393,0.010258478845952476,-1.8261599971473292>, <-0.18059834733211774,-0.09177549654042169,-0.5969504716961467>, 0.5}
    cylinder { m*<-1.8022137072250077,2.2366984478781777,-1.5708962371121158>, <-0.18059834733211774,-0.09177549654042169,-0.5969504716961467>, 0.5 }
    cylinder {  m*<-1.534426486187176,-2.6509934945257196,-1.3813499519495431>, <-0.18059834733211774,-0.09177549654042169,-0.5969504716961467>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    