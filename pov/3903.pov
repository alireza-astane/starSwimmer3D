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
    sphere { m*<-0.10112177099471487,0.016722192289152726,-0.18663977313201158>, 1 }        
    sphere {  m*<0.1396133337469767,0.14543227046947793,2.800914997988539>, 1 }
    sphere {  m*<2.6335866230115474,0.1187561676755271,-1.4158492985831974>, 1 }
    sphere {  m*<-1.7227371308876065,2.345196136707755,-1.1605855385479833>, 1}
    sphere { m*<-1.7582074129532956,-3.115760637244838,-1.1467447369621606>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1396133337469767,0.14543227046947793,2.800914997988539>, <-0.10112177099471487,0.016722192289152726,-0.18663977313201158>, 0.5 }
    cylinder { m*<2.6335866230115474,0.1187561676755271,-1.4158492985831974>, <-0.10112177099471487,0.016722192289152726,-0.18663977313201158>, 0.5}
    cylinder { m*<-1.7227371308876065,2.345196136707755,-1.1605855385479833>, <-0.10112177099471487,0.016722192289152726,-0.18663977313201158>, 0.5 }
    cylinder {  m*<-1.7582074129532956,-3.115760637244838,-1.1467447369621606>, <-0.10112177099471487,0.016722192289152726,-0.18663977313201158>, 0.5}

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
    sphere { m*<-0.10112177099471487,0.016722192289152726,-0.18663977313201158>, 1 }        
    sphere {  m*<0.1396133337469767,0.14543227046947793,2.800914997988539>, 1 }
    sphere {  m*<2.6335866230115474,0.1187561676755271,-1.4158492985831974>, 1 }
    sphere {  m*<-1.7227371308876065,2.345196136707755,-1.1605855385479833>, 1}
    sphere { m*<-1.7582074129532956,-3.115760637244838,-1.1467447369621606>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1396133337469767,0.14543227046947793,2.800914997988539>, <-0.10112177099471487,0.016722192289152726,-0.18663977313201158>, 0.5 }
    cylinder { m*<2.6335866230115474,0.1187561676755271,-1.4158492985831974>, <-0.10112177099471487,0.016722192289152726,-0.18663977313201158>, 0.5}
    cylinder { m*<-1.7227371308876065,2.345196136707755,-1.1605855385479833>, <-0.10112177099471487,0.016722192289152726,-0.18663977313201158>, 0.5 }
    cylinder {  m*<-1.7582074129532956,-3.115760637244838,-1.1467447369621606>, <-0.10112177099471487,0.016722192289152726,-0.18663977313201158>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    