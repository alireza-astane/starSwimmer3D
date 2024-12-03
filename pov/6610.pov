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
    sphere { m*<-1.1489184041185774,-0.8525408391485566,-0.7984378645511426>, 1 }        
    sphere {  m*<0.29143989531543113,-0.10621612970189985,9.06917902928602>, 1 }
    sphere {  m*<7.646791333315398,-0.19513640569625668,-5.510314260759319>, 1 }
    sphere {  m*<-5.5038608506777456,4.53419573420706,-3.028518978403948>, 1}
    sphere { m*<-2.4242157529060084,-3.494222409286946,-1.4257098890294477>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.29143989531543113,-0.10621612970189985,9.06917902928602>, <-1.1489184041185774,-0.8525408391485566,-0.7984378645511426>, 0.5 }
    cylinder { m*<7.646791333315398,-0.19513640569625668,-5.510314260759319>, <-1.1489184041185774,-0.8525408391485566,-0.7984378645511426>, 0.5}
    cylinder { m*<-5.5038608506777456,4.53419573420706,-3.028518978403948>, <-1.1489184041185774,-0.8525408391485566,-0.7984378645511426>, 0.5 }
    cylinder {  m*<-2.4242157529060084,-3.494222409286946,-1.4257098890294477>, <-1.1489184041185774,-0.8525408391485566,-0.7984378645511426>, 0.5}

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
    sphere { m*<-1.1489184041185774,-0.8525408391485566,-0.7984378645511426>, 1 }        
    sphere {  m*<0.29143989531543113,-0.10621612970189985,9.06917902928602>, 1 }
    sphere {  m*<7.646791333315398,-0.19513640569625668,-5.510314260759319>, 1 }
    sphere {  m*<-5.5038608506777456,4.53419573420706,-3.028518978403948>, 1}
    sphere { m*<-2.4242157529060084,-3.494222409286946,-1.4257098890294477>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.29143989531543113,-0.10621612970189985,9.06917902928602>, <-1.1489184041185774,-0.8525408391485566,-0.7984378645511426>, 0.5 }
    cylinder { m*<7.646791333315398,-0.19513640569625668,-5.510314260759319>, <-1.1489184041185774,-0.8525408391485566,-0.7984378645511426>, 0.5}
    cylinder { m*<-5.5038608506777456,4.53419573420706,-3.028518978403948>, <-1.1489184041185774,-0.8525408391485566,-0.7984378645511426>, 0.5 }
    cylinder {  m*<-2.4242157529060084,-3.494222409286946,-1.4257098890294477>, <-1.1489184041185774,-0.8525408391485566,-0.7984378645511426>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    