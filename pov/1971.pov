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
    sphere { m*<1.2498605816600887,-6.545877722521212e-19,0.640047754740587>, 1 }        
    sphere {  m*<1.4859460675058116,8.846652408385808e-20,3.630753772092244>, 1 }
    sphere {  m*<4.169808248354892,6.2594529441497344e-18,-0.6594062854746294>, 1 }
    sphere {  m*<-3.7118977479568693,8.164965809277259,-2.3119863801362968>, 1}
    sphere { m*<-3.7118977479568693,-8.164965809277259,-2.3119863801363003>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4859460675058116,8.846652408385808e-20,3.630753772092244>, <1.2498605816600887,-6.545877722521212e-19,0.640047754740587>, 0.5 }
    cylinder { m*<4.169808248354892,6.2594529441497344e-18,-0.6594062854746294>, <1.2498605816600887,-6.545877722521212e-19,0.640047754740587>, 0.5}
    cylinder { m*<-3.7118977479568693,8.164965809277259,-2.3119863801362968>, <1.2498605816600887,-6.545877722521212e-19,0.640047754740587>, 0.5 }
    cylinder {  m*<-3.7118977479568693,-8.164965809277259,-2.3119863801363003>, <1.2498605816600887,-6.545877722521212e-19,0.640047754740587>, 0.5}

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
    sphere { m*<1.2498605816600887,-6.545877722521212e-19,0.640047754740587>, 1 }        
    sphere {  m*<1.4859460675058116,8.846652408385808e-20,3.630753772092244>, 1 }
    sphere {  m*<4.169808248354892,6.2594529441497344e-18,-0.6594062854746294>, 1 }
    sphere {  m*<-3.7118977479568693,8.164965809277259,-2.3119863801362968>, 1}
    sphere { m*<-3.7118977479568693,-8.164965809277259,-2.3119863801363003>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4859460675058116,8.846652408385808e-20,3.630753772092244>, <1.2498605816600887,-6.545877722521212e-19,0.640047754740587>, 0.5 }
    cylinder { m*<4.169808248354892,6.2594529441497344e-18,-0.6594062854746294>, <1.2498605816600887,-6.545877722521212e-19,0.640047754740587>, 0.5}
    cylinder { m*<-3.7118977479568693,8.164965809277259,-2.3119863801362968>, <1.2498605816600887,-6.545877722521212e-19,0.640047754740587>, 0.5 }
    cylinder {  m*<-3.7118977479568693,-8.164965809277259,-2.3119863801363003>, <1.2498605816600887,-6.545877722521212e-19,0.640047754740587>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    