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
    sphere { m*<0.8843495477046773,-2.3355372966248487e-18,0.8370799451617916>, 1 }        
    sphere {  m*<1.0309703001823955,1.334550816346304e-18,3.8335003928235194>, 1 }
    sphere {  m*<5.869353839283108,4.753791356612042e-18,-1.2049062368922174>, 1 }
    sphere {  m*<-3.986982883554207,8.164965809277259,-2.261883822596424>, 1}
    sphere { m*<-3.986982883554207,-8.164965809277259,-2.2618838225964275>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0309703001823955,1.334550816346304e-18,3.8335003928235194>, <0.8843495477046773,-2.3355372966248487e-18,0.8370799451617916>, 0.5 }
    cylinder { m*<5.869353839283108,4.753791356612042e-18,-1.2049062368922174>, <0.8843495477046773,-2.3355372966248487e-18,0.8370799451617916>, 0.5}
    cylinder { m*<-3.986982883554207,8.164965809277259,-2.261883822596424>, <0.8843495477046773,-2.3355372966248487e-18,0.8370799451617916>, 0.5 }
    cylinder {  m*<-3.986982883554207,-8.164965809277259,-2.2618838225964275>, <0.8843495477046773,-2.3355372966248487e-18,0.8370799451617916>, 0.5}

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
    sphere { m*<0.8843495477046773,-2.3355372966248487e-18,0.8370799451617916>, 1 }        
    sphere {  m*<1.0309703001823955,1.334550816346304e-18,3.8335003928235194>, 1 }
    sphere {  m*<5.869353839283108,4.753791356612042e-18,-1.2049062368922174>, 1 }
    sphere {  m*<-3.986982883554207,8.164965809277259,-2.261883822596424>, 1}
    sphere { m*<-3.986982883554207,-8.164965809277259,-2.2618838225964275>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0309703001823955,1.334550816346304e-18,3.8335003928235194>, <0.8843495477046773,-2.3355372966248487e-18,0.8370799451617916>, 0.5 }
    cylinder { m*<5.869353839283108,4.753791356612042e-18,-1.2049062368922174>, <0.8843495477046773,-2.3355372966248487e-18,0.8370799451617916>, 0.5}
    cylinder { m*<-3.986982883554207,8.164965809277259,-2.261883822596424>, <0.8843495477046773,-2.3355372966248487e-18,0.8370799451617916>, 0.5 }
    cylinder {  m*<-3.986982883554207,-8.164965809277259,-2.2618838225964275>, <0.8843495477046773,-2.3355372966248487e-18,0.8370799451617916>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    