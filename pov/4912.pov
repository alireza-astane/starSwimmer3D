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
    sphere { m*<-0.25573843835588617,-0.13194947532313084,-1.52944902724069>, 1 }        
    sphere {  m*<0.4978426845205214,0.27095597274068006,7.8225932579569974>, 1 }
    sphere {  m*<2.478969955650371,-0.02991549993675667,-2.758658552691872>, 1 }
    sphere {  m*<-1.8773537982487762,2.1965244690954684,-2.5033947926566587>, 1}
    sphere { m*<-1.6095665772109444,-2.691167473308429,-2.3138485074940855>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4978426845205214,0.27095597274068006,7.8225932579569974>, <-0.25573843835588617,-0.13194947532313084,-1.52944902724069>, 0.5 }
    cylinder { m*<2.478969955650371,-0.02991549993675667,-2.758658552691872>, <-0.25573843835588617,-0.13194947532313084,-1.52944902724069>, 0.5}
    cylinder { m*<-1.8773537982487762,2.1965244690954684,-2.5033947926566587>, <-0.25573843835588617,-0.13194947532313084,-1.52944902724069>, 0.5 }
    cylinder {  m*<-1.6095665772109444,-2.691167473308429,-2.3138485074940855>, <-0.25573843835588617,-0.13194947532313084,-1.52944902724069>, 0.5}

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
    sphere { m*<-0.25573843835588617,-0.13194947532313084,-1.52944902724069>, 1 }        
    sphere {  m*<0.4978426845205214,0.27095597274068006,7.8225932579569974>, 1 }
    sphere {  m*<2.478969955650371,-0.02991549993675667,-2.758658552691872>, 1 }
    sphere {  m*<-1.8773537982487762,2.1965244690954684,-2.5033947926566587>, 1}
    sphere { m*<-1.6095665772109444,-2.691167473308429,-2.3138485074940855>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4978426845205214,0.27095597274068006,7.8225932579569974>, <-0.25573843835588617,-0.13194947532313084,-1.52944902724069>, 0.5 }
    cylinder { m*<2.478969955650371,-0.02991549993675667,-2.758658552691872>, <-0.25573843835588617,-0.13194947532313084,-1.52944902724069>, 0.5}
    cylinder { m*<-1.8773537982487762,2.1965244690954684,-2.5033947926566587>, <-0.25573843835588617,-0.13194947532313084,-1.52944902724069>, 0.5 }
    cylinder {  m*<-1.6095665772109444,-2.691167473308429,-2.3138485074940855>, <-0.25573843835588617,-0.13194947532313084,-1.52944902724069>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    