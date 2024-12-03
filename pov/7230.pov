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
    sphere { m*<-0.7023473030996635,-1.0501155196663847,-0.5748122972779157>, 1 }        
    sphere {  m*<0.7168201911004991,-0.060176605786466775,9.274477799757237>, 1 }
    sphere {  m*<8.084607389423295,-0.34526885657872985,-5.296199629316696>, 1 }
    sphere {  m*<-6.811355804265691,6.177812517041925,-3.805392726135093>, 1}
    sphere { m*<-2.5939846917213782,-5.169732395100543,-1.450806019873382>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7168201911004991,-0.060176605786466775,9.274477799757237>, <-0.7023473030996635,-1.0501155196663847,-0.5748122972779157>, 0.5 }
    cylinder { m*<8.084607389423295,-0.34526885657872985,-5.296199629316696>, <-0.7023473030996635,-1.0501155196663847,-0.5748122972779157>, 0.5}
    cylinder { m*<-6.811355804265691,6.177812517041925,-3.805392726135093>, <-0.7023473030996635,-1.0501155196663847,-0.5748122972779157>, 0.5 }
    cylinder {  m*<-2.5939846917213782,-5.169732395100543,-1.450806019873382>, <-0.7023473030996635,-1.0501155196663847,-0.5748122972779157>, 0.5}

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
    sphere { m*<-0.7023473030996635,-1.0501155196663847,-0.5748122972779157>, 1 }        
    sphere {  m*<0.7168201911004991,-0.060176605786466775,9.274477799757237>, 1 }
    sphere {  m*<8.084607389423295,-0.34526885657872985,-5.296199629316696>, 1 }
    sphere {  m*<-6.811355804265691,6.177812517041925,-3.805392726135093>, 1}
    sphere { m*<-2.5939846917213782,-5.169732395100543,-1.450806019873382>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7168201911004991,-0.060176605786466775,9.274477799757237>, <-0.7023473030996635,-1.0501155196663847,-0.5748122972779157>, 0.5 }
    cylinder { m*<8.084607389423295,-0.34526885657872985,-5.296199629316696>, <-0.7023473030996635,-1.0501155196663847,-0.5748122972779157>, 0.5}
    cylinder { m*<-6.811355804265691,6.177812517041925,-3.805392726135093>, <-0.7023473030996635,-1.0501155196663847,-0.5748122972779157>, 0.5 }
    cylinder {  m*<-2.5939846917213782,-5.169732395100543,-1.450806019873382>, <-0.7023473030996635,-1.0501155196663847,-0.5748122972779157>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    