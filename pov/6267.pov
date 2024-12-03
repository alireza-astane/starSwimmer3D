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
    sphere { m*<-1.4172411832418865,-0.4744944481633411,-0.9361227560764086>, 1 }        
    sphere {  m*<0.03887443665599033,0.10366371467748753,8.940471705417435>, 1 }
    sphere {  m*<7.394225874655965,0.01474343868313055,-5.639021584627917>, 1 }
    sphere {  m*<-4.239521001616308,3.2292831350380746,-2.382657812461665>, 1}
    sphere { m*<-2.760988011132455,-3.0734028504116764,-1.5981904139148864>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.03887443665599033,0.10366371467748753,8.940471705417435>, <-1.4172411832418865,-0.4744944481633411,-0.9361227560764086>, 0.5 }
    cylinder { m*<7.394225874655965,0.01474343868313055,-5.639021584627917>, <-1.4172411832418865,-0.4744944481633411,-0.9361227560764086>, 0.5}
    cylinder { m*<-4.239521001616308,3.2292831350380746,-2.382657812461665>, <-1.4172411832418865,-0.4744944481633411,-0.9361227560764086>, 0.5 }
    cylinder {  m*<-2.760988011132455,-3.0734028504116764,-1.5981904139148864>, <-1.4172411832418865,-0.4744944481633411,-0.9361227560764086>, 0.5}

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
    sphere { m*<-1.4172411832418865,-0.4744944481633411,-0.9361227560764086>, 1 }        
    sphere {  m*<0.03887443665599033,0.10366371467748753,8.940471705417435>, 1 }
    sphere {  m*<7.394225874655965,0.01474343868313055,-5.639021584627917>, 1 }
    sphere {  m*<-4.239521001616308,3.2292831350380746,-2.382657812461665>, 1}
    sphere { m*<-2.760988011132455,-3.0734028504116764,-1.5981904139148864>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.03887443665599033,0.10366371467748753,8.940471705417435>, <-1.4172411832418865,-0.4744944481633411,-0.9361227560764086>, 0.5 }
    cylinder { m*<7.394225874655965,0.01474343868313055,-5.639021584627917>, <-1.4172411832418865,-0.4744944481633411,-0.9361227560764086>, 0.5}
    cylinder { m*<-4.239521001616308,3.2292831350380746,-2.382657812461665>, <-1.4172411832418865,-0.4744944481633411,-0.9361227560764086>, 0.5 }
    cylinder {  m*<-2.760988011132455,-3.0734028504116764,-1.5981904139148864>, <-1.4172411832418865,-0.4744944481633411,-0.9361227560764086>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    