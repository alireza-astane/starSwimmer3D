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
    sphere { m*<0.6380029865105056,-6.072929053629943e-18,0.9500727177061781>, 1 }        
    sphere {  m*<0.7353722223434808,-2.589433461475936e-18,3.9484955878618324>, 1 }
    sphere {  m*<6.907828573462779,2.0495245744014996e-18,-1.498560163530173>, 1 }
    sphere {  m*<-4.181752040752567,8.164965809277259,-2.2285072559413326>, 1}
    sphere { m*<-4.181752040752567,-8.164965809277259,-2.2285072559413353>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7353722223434808,-2.589433461475936e-18,3.9484955878618324>, <0.6380029865105056,-6.072929053629943e-18,0.9500727177061781>, 0.5 }
    cylinder { m*<6.907828573462779,2.0495245744014996e-18,-1.498560163530173>, <0.6380029865105056,-6.072929053629943e-18,0.9500727177061781>, 0.5}
    cylinder { m*<-4.181752040752567,8.164965809277259,-2.2285072559413326>, <0.6380029865105056,-6.072929053629943e-18,0.9500727177061781>, 0.5 }
    cylinder {  m*<-4.181752040752567,-8.164965809277259,-2.2285072559413353>, <0.6380029865105056,-6.072929053629943e-18,0.9500727177061781>, 0.5}

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
    sphere { m*<0.6380029865105056,-6.072929053629943e-18,0.9500727177061781>, 1 }        
    sphere {  m*<0.7353722223434808,-2.589433461475936e-18,3.9484955878618324>, 1 }
    sphere {  m*<6.907828573462779,2.0495245744014996e-18,-1.498560163530173>, 1 }
    sphere {  m*<-4.181752040752567,8.164965809277259,-2.2285072559413326>, 1}
    sphere { m*<-4.181752040752567,-8.164965809277259,-2.2285072559413353>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7353722223434808,-2.589433461475936e-18,3.9484955878618324>, <0.6380029865105056,-6.072929053629943e-18,0.9500727177061781>, 0.5 }
    cylinder { m*<6.907828573462779,2.0495245744014996e-18,-1.498560163530173>, <0.6380029865105056,-6.072929053629943e-18,0.9500727177061781>, 0.5}
    cylinder { m*<-4.181752040752567,8.164965809277259,-2.2285072559413326>, <0.6380029865105056,-6.072929053629943e-18,0.9500727177061781>, 0.5 }
    cylinder {  m*<-4.181752040752567,-8.164965809277259,-2.2285072559413353>, <0.6380029865105056,-6.072929053629943e-18,0.9500727177061781>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    