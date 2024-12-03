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
    sphere { m*<0.21383683326897485,0.6121062779927051,-0.004154745383432662>, 1 }        
    sphere {  m*<0.45457193801066653,0.7408163561730307,2.9834000257371183>, 1 }
    sphere {  m*<2.9485452272752326,0.7141402533790795,-1.2333642708346169>, 1 }
    sphere {  m*<-1.407778526623916,2.940580222411306,-0.9781005107994026>, 1}
    sphere { m*<-3.016397106039191,-5.494187865200205,-1.875731905268029>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.45457193801066653,0.7408163561730307,2.9834000257371183>, <0.21383683326897485,0.6121062779927051,-0.004154745383432662>, 0.5 }
    cylinder { m*<2.9485452272752326,0.7141402533790795,-1.2333642708346169>, <0.21383683326897485,0.6121062779927051,-0.004154745383432662>, 0.5}
    cylinder { m*<-1.407778526623916,2.940580222411306,-0.9781005107994026>, <0.21383683326897485,0.6121062779927051,-0.004154745383432662>, 0.5 }
    cylinder {  m*<-3.016397106039191,-5.494187865200205,-1.875731905268029>, <0.21383683326897485,0.6121062779927051,-0.004154745383432662>, 0.5}

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
    sphere { m*<0.21383683326897485,0.6121062779927051,-0.004154745383432662>, 1 }        
    sphere {  m*<0.45457193801066653,0.7408163561730307,2.9834000257371183>, 1 }
    sphere {  m*<2.9485452272752326,0.7141402533790795,-1.2333642708346169>, 1 }
    sphere {  m*<-1.407778526623916,2.940580222411306,-0.9781005107994026>, 1}
    sphere { m*<-3.016397106039191,-5.494187865200205,-1.875731905268029>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.45457193801066653,0.7408163561730307,2.9834000257371183>, <0.21383683326897485,0.6121062779927051,-0.004154745383432662>, 0.5 }
    cylinder { m*<2.9485452272752326,0.7141402533790795,-1.2333642708346169>, <0.21383683326897485,0.6121062779927051,-0.004154745383432662>, 0.5}
    cylinder { m*<-1.407778526623916,2.940580222411306,-0.9781005107994026>, <0.21383683326897485,0.6121062779927051,-0.004154745383432662>, 0.5 }
    cylinder {  m*<-3.016397106039191,-5.494187865200205,-1.875731905268029>, <0.21383683326897485,0.6121062779927051,-0.004154745383432662>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    