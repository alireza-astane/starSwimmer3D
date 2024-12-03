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
    sphere { m*<-1.4942844259268921,-0.18235035607502154,-1.0810761475629902>, 1 }        
    sphere {  m*<-0.0673855163333574,0.27806807687597135,8.805841511136398>, 1 }
    sphere {  m*<6.920407477659875,0.10629867017832362,-5.517763772264088>, 1 }
    sphere {  m*<-3.1684218401798523,2.1469450881912087,-1.9595033145019718>, 1}
    sphere { m*<-2.900634619142021,-2.7407468542126887,-1.7699570293394014>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.0673855163333574,0.27806807687597135,8.805841511136398>, <-1.4942844259268921,-0.18235035607502154,-1.0810761475629902>, 0.5 }
    cylinder { m*<6.920407477659875,0.10629867017832362,-5.517763772264088>, <-1.4942844259268921,-0.18235035607502154,-1.0810761475629902>, 0.5}
    cylinder { m*<-3.1684218401798523,2.1469450881912087,-1.9595033145019718>, <-1.4942844259268921,-0.18235035607502154,-1.0810761475629902>, 0.5 }
    cylinder {  m*<-2.900634619142021,-2.7407468542126887,-1.7699570293394014>, <-1.4942844259268921,-0.18235035607502154,-1.0810761475629902>, 0.5}

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
    sphere { m*<-1.4942844259268921,-0.18235035607502154,-1.0810761475629902>, 1 }        
    sphere {  m*<-0.0673855163333574,0.27806807687597135,8.805841511136398>, 1 }
    sphere {  m*<6.920407477659875,0.10629867017832362,-5.517763772264088>, 1 }
    sphere {  m*<-3.1684218401798523,2.1469450881912087,-1.9595033145019718>, 1}
    sphere { m*<-2.900634619142021,-2.7407468542126887,-1.7699570293394014>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.0673855163333574,0.27806807687597135,8.805841511136398>, <-1.4942844259268921,-0.18235035607502154,-1.0810761475629902>, 0.5 }
    cylinder { m*<6.920407477659875,0.10629867017832362,-5.517763772264088>, <-1.4942844259268921,-0.18235035607502154,-1.0810761475629902>, 0.5}
    cylinder { m*<-3.1684218401798523,2.1469450881912087,-1.9595033145019718>, <-1.4942844259268921,-0.18235035607502154,-1.0810761475629902>, 0.5 }
    cylinder {  m*<-2.900634619142021,-2.7407468542126887,-1.7699570293394014>, <-1.4942844259268921,-0.18235035607502154,-1.0810761475629902>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    