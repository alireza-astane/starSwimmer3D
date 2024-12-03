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
    sphere { m*<0.9203862370311483,-1.580560303584449e-18,0.8193803661624997>, 1 }        
    sphere {  m*<1.0749002545794761,1.5997852719008607e-18,3.815404517676904>, 1 }
    sphere {  m*<5.712340777734323,5.664817369886213e-18,-1.1583053313406915>, 1 }
    sphere {  m*<-3.959093769432569,8.164965809277259,-2.266740003082843>, 1}
    sphere { m*<-3.959093769432569,-8.164965809277259,-2.2667400030828455>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0749002545794761,1.5997852719008607e-18,3.815404517676904>, <0.9203862370311483,-1.580560303584449e-18,0.8193803661624997>, 0.5 }
    cylinder { m*<5.712340777734323,5.664817369886213e-18,-1.1583053313406915>, <0.9203862370311483,-1.580560303584449e-18,0.8193803661624997>, 0.5}
    cylinder { m*<-3.959093769432569,8.164965809277259,-2.266740003082843>, <0.9203862370311483,-1.580560303584449e-18,0.8193803661624997>, 0.5 }
    cylinder {  m*<-3.959093769432569,-8.164965809277259,-2.2667400030828455>, <0.9203862370311483,-1.580560303584449e-18,0.8193803661624997>, 0.5}

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
    sphere { m*<0.9203862370311483,-1.580560303584449e-18,0.8193803661624997>, 1 }        
    sphere {  m*<1.0749002545794761,1.5997852719008607e-18,3.815404517676904>, 1 }
    sphere {  m*<5.712340777734323,5.664817369886213e-18,-1.1583053313406915>, 1 }
    sphere {  m*<-3.959093769432569,8.164965809277259,-2.266740003082843>, 1}
    sphere { m*<-3.959093769432569,-8.164965809277259,-2.2667400030828455>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0749002545794761,1.5997852719008607e-18,3.815404517676904>, <0.9203862370311483,-1.580560303584449e-18,0.8193803661624997>, 0.5 }
    cylinder { m*<5.712340777734323,5.664817369886213e-18,-1.1583053313406915>, <0.9203862370311483,-1.580560303584449e-18,0.8193803661624997>, 0.5}
    cylinder { m*<-3.959093769432569,8.164965809277259,-2.266740003082843>, <0.9203862370311483,-1.580560303584449e-18,0.8193803661624997>, 0.5 }
    cylinder {  m*<-3.959093769432569,-8.164965809277259,-2.2667400030828455>, <0.9203862370311483,-1.580560303584449e-18,0.8193803661624997>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    