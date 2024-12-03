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
    sphere { m*<-0.19722639823032484,-0.10066575678473083,-0.8033068020231156>, 1 }        
    sphere {  m*<0.2979657122233344,0.164090874032198,5.3420933621718545>, 1 }
    sphere {  m*<2.5374819957759325,0.0013682186016433645,-2.032516327474297>, 1 }
    sphere {  m*<-1.8188417581232148,2.227808187633868,-1.7772525674390838>, 1}
    sphere { m*<-1.551054537085383,-2.6598837547700294,-1.587706282276511>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2979657122233344,0.164090874032198,5.3420933621718545>, <-0.19722639823032484,-0.10066575678473083,-0.8033068020231156>, 0.5 }
    cylinder { m*<2.5374819957759325,0.0013682186016433645,-2.032516327474297>, <-0.19722639823032484,-0.10066575678473083,-0.8033068020231156>, 0.5}
    cylinder { m*<-1.8188417581232148,2.227808187633868,-1.7772525674390838>, <-0.19722639823032484,-0.10066575678473083,-0.8033068020231156>, 0.5 }
    cylinder {  m*<-1.551054537085383,-2.6598837547700294,-1.587706282276511>, <-0.19722639823032484,-0.10066575678473083,-0.8033068020231156>, 0.5}

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
    sphere { m*<-0.19722639823032484,-0.10066575678473083,-0.8033068020231156>, 1 }        
    sphere {  m*<0.2979657122233344,0.164090874032198,5.3420933621718545>, 1 }
    sphere {  m*<2.5374819957759325,0.0013682186016433645,-2.032516327474297>, 1 }
    sphere {  m*<-1.8188417581232148,2.227808187633868,-1.7772525674390838>, 1}
    sphere { m*<-1.551054537085383,-2.6598837547700294,-1.587706282276511>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2979657122233344,0.164090874032198,5.3420933621718545>, <-0.19722639823032484,-0.10066575678473083,-0.8033068020231156>, 0.5 }
    cylinder { m*<2.5374819957759325,0.0013682186016433645,-2.032516327474297>, <-0.19722639823032484,-0.10066575678473083,-0.8033068020231156>, 0.5}
    cylinder { m*<-1.8188417581232148,2.227808187633868,-1.7772525674390838>, <-0.19722639823032484,-0.10066575678473083,-0.8033068020231156>, 0.5 }
    cylinder {  m*<-1.551054537085383,-2.6598837547700294,-1.587706282276511>, <-0.19722639823032484,-0.10066575678473083,-0.8033068020231156>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    