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
    sphere { m*<0.5780220084485562,1.0898542958215567,0.20763527585127128>, 1 }        
    sphere {  m*<0.8195350238709538,1.1947173678980536,3.19605475334356>, 1 }
    sphere {  m*<3.3127822129334876,1.1947173678980532,-1.021227455147055>, 1 }
    sphere {  m*<-1.3924185579643429,3.9431172237079393,-0.9574053906939972>, 1}
    sphere { m*<-3.9567146677284586,-7.409684712296257,-2.4729300890802026>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8195350238709538,1.1947173678980536,3.19605475334356>, <0.5780220084485562,1.0898542958215567,0.20763527585127128>, 0.5 }
    cylinder { m*<3.3127822129334876,1.1947173678980532,-1.021227455147055>, <0.5780220084485562,1.0898542958215567,0.20763527585127128>, 0.5}
    cylinder { m*<-1.3924185579643429,3.9431172237079393,-0.9574053906939972>, <0.5780220084485562,1.0898542958215567,0.20763527585127128>, 0.5 }
    cylinder {  m*<-3.9567146677284586,-7.409684712296257,-2.4729300890802026>, <0.5780220084485562,1.0898542958215567,0.20763527585127128>, 0.5}

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
    sphere { m*<0.5780220084485562,1.0898542958215567,0.20763527585127128>, 1 }        
    sphere {  m*<0.8195350238709538,1.1947173678980536,3.19605475334356>, 1 }
    sphere {  m*<3.3127822129334876,1.1947173678980532,-1.021227455147055>, 1 }
    sphere {  m*<-1.3924185579643429,3.9431172237079393,-0.9574053906939972>, 1}
    sphere { m*<-3.9567146677284586,-7.409684712296257,-2.4729300890802026>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8195350238709538,1.1947173678980536,3.19605475334356>, <0.5780220084485562,1.0898542958215567,0.20763527585127128>, 0.5 }
    cylinder { m*<3.3127822129334876,1.1947173678980532,-1.021227455147055>, <0.5780220084485562,1.0898542958215567,0.20763527585127128>, 0.5}
    cylinder { m*<-1.3924185579643429,3.9431172237079393,-0.9574053906939972>, <0.5780220084485562,1.0898542958215567,0.20763527585127128>, 0.5 }
    cylinder {  m*<-3.9567146677284586,-7.409684712296257,-2.4729300890802026>, <0.5780220084485562,1.0898542958215567,0.20763527585127128>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    