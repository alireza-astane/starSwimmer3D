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
    sphere { m*<-1.392077976229944,-0.17898204242381244,-1.1385247400719598>, 1 }        
    sphere {  m*<-0.007994234475360962,0.27933386025309226,8.7545784153504>, 1 }
    sphere {  m*<6.5888059756261,0.09648711336624755,-5.302501977434677>, 1 }
    sphere {  m*<-3.0627945920542734,2.1502422900047864,-2.0236283572228166>, 1}
    sphere { m*<-2.795007371016442,-2.737449652399111,-1.8340820720602464>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.007994234475360962,0.27933386025309226,8.7545784153504>, <-1.392077976229944,-0.17898204242381244,-1.1385247400719598>, 0.5 }
    cylinder { m*<6.5888059756261,0.09648711336624755,-5.302501977434677>, <-1.392077976229944,-0.17898204242381244,-1.1385247400719598>, 0.5}
    cylinder { m*<-3.0627945920542734,2.1502422900047864,-2.0236283572228166>, <-1.392077976229944,-0.17898204242381244,-1.1385247400719598>, 0.5 }
    cylinder {  m*<-2.795007371016442,-2.737449652399111,-1.8340820720602464>, <-1.392077976229944,-0.17898204242381244,-1.1385247400719598>, 0.5}

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
    sphere { m*<-1.392077976229944,-0.17898204242381244,-1.1385247400719598>, 1 }        
    sphere {  m*<-0.007994234475360962,0.27933386025309226,8.7545784153504>, 1 }
    sphere {  m*<6.5888059756261,0.09648711336624755,-5.302501977434677>, 1 }
    sphere {  m*<-3.0627945920542734,2.1502422900047864,-2.0236283572228166>, 1}
    sphere { m*<-2.795007371016442,-2.737449652399111,-1.8340820720602464>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.007994234475360962,0.27933386025309226,8.7545784153504>, <-1.392077976229944,-0.17898204242381244,-1.1385247400719598>, 0.5 }
    cylinder { m*<6.5888059756261,0.09648711336624755,-5.302501977434677>, <-1.392077976229944,-0.17898204242381244,-1.1385247400719598>, 0.5}
    cylinder { m*<-3.0627945920542734,2.1502422900047864,-2.0236283572228166>, <-1.392077976229944,-0.17898204242381244,-1.1385247400719598>, 0.5 }
    cylinder {  m*<-2.795007371016442,-2.737449652399111,-1.8340820720602464>, <-1.392077976229944,-0.17898204242381244,-1.1385247400719598>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    