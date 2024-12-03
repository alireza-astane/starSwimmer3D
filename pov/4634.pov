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
    sphere { m*<-0.21957441762321167,-0.11261422323843015,-1.080648704584246>, 1 }        
    sphere {  m*<0.37784986731075265,0.2068012874457435,6.333466385746585>, 1 }
    sphere {  m*<2.5151339763830456,-0.010580247852055913,-2.3098582300354265>, 1 }
    sphere {  m*<-1.8411897775161017,2.215859721180169,-2.0545944700002132>, 1}
    sphere { m*<-1.5734025564782699,-2.6718322212237284,-1.8650481848376406>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.37784986731075265,0.2068012874457435,6.333466385746585>, <-0.21957441762321167,-0.11261422323843015,-1.080648704584246>, 0.5 }
    cylinder { m*<2.5151339763830456,-0.010580247852055913,-2.3098582300354265>, <-0.21957441762321167,-0.11261422323843015,-1.080648704584246>, 0.5}
    cylinder { m*<-1.8411897775161017,2.215859721180169,-2.0545944700002132>, <-0.21957441762321167,-0.11261422323843015,-1.080648704584246>, 0.5 }
    cylinder {  m*<-1.5734025564782699,-2.6718322212237284,-1.8650481848376406>, <-0.21957441762321167,-0.11261422323843015,-1.080648704584246>, 0.5}

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
    sphere { m*<-0.21957441762321167,-0.11261422323843015,-1.080648704584246>, 1 }        
    sphere {  m*<0.37784986731075265,0.2068012874457435,6.333466385746585>, 1 }
    sphere {  m*<2.5151339763830456,-0.010580247852055913,-2.3098582300354265>, 1 }
    sphere {  m*<-1.8411897775161017,2.215859721180169,-2.0545944700002132>, 1}
    sphere { m*<-1.5734025564782699,-2.6718322212237284,-1.8650481848376406>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.37784986731075265,0.2068012874457435,6.333466385746585>, <-0.21957441762321167,-0.11261422323843015,-1.080648704584246>, 0.5 }
    cylinder { m*<2.5151339763830456,-0.010580247852055913,-2.3098582300354265>, <-0.21957441762321167,-0.11261422323843015,-1.080648704584246>, 0.5}
    cylinder { m*<-1.8411897775161017,2.215859721180169,-2.0545944700002132>, <-0.21957441762321167,-0.11261422323843015,-1.080648704584246>, 0.5 }
    cylinder {  m*<-1.5734025564782699,-2.6718322212237284,-1.8650481848376406>, <-0.21957441762321167,-0.11261422323843015,-1.080648704584246>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    