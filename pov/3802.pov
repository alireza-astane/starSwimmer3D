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
    sphere { m*<-0.04447016867513265,0.12381392432656724,-0.15381619225077564>, 1 }        
    sphere {  m*<0.19626493606655887,0.2525240025068928,2.833738578869775>, 1 }
    sphere {  m*<2.6902382253311306,0.22584789971294195,-1.3830257177019623>, 1 }
    sphere {  m*<-1.6660855285680238,2.45228786874517,-1.1277619576667477>, 1}
    sphere { m*<-2.0206079753572226,-3.611791280065958,-1.2987779669273087>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19626493606655887,0.2525240025068928,2.833738578869775>, <-0.04447016867513265,0.12381392432656724,-0.15381619225077564>, 0.5 }
    cylinder { m*<2.6902382253311306,0.22584789971294195,-1.3830257177019623>, <-0.04447016867513265,0.12381392432656724,-0.15381619225077564>, 0.5}
    cylinder { m*<-1.6660855285680238,2.45228786874517,-1.1277619576667477>, <-0.04447016867513265,0.12381392432656724,-0.15381619225077564>, 0.5 }
    cylinder {  m*<-2.0206079753572226,-3.611791280065958,-1.2987779669273087>, <-0.04447016867513265,0.12381392432656724,-0.15381619225077564>, 0.5}

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
    sphere { m*<-0.04447016867513265,0.12381392432656724,-0.15381619225077564>, 1 }        
    sphere {  m*<0.19626493606655887,0.2525240025068928,2.833738578869775>, 1 }
    sphere {  m*<2.6902382253311306,0.22584789971294195,-1.3830257177019623>, 1 }
    sphere {  m*<-1.6660855285680238,2.45228786874517,-1.1277619576667477>, 1}
    sphere { m*<-2.0206079753572226,-3.611791280065958,-1.2987779669273087>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19626493606655887,0.2525240025068928,2.833738578869775>, <-0.04447016867513265,0.12381392432656724,-0.15381619225077564>, 0.5 }
    cylinder { m*<2.6902382253311306,0.22584789971294195,-1.3830257177019623>, <-0.04447016867513265,0.12381392432656724,-0.15381619225077564>, 0.5}
    cylinder { m*<-1.6660855285680238,2.45228786874517,-1.1277619576667477>, <-0.04447016867513265,0.12381392432656724,-0.15381619225077564>, 0.5 }
    cylinder {  m*<-2.0206079753572226,-3.611791280065958,-1.2987779669273087>, <-0.04447016867513265,0.12381392432656724,-0.15381619225077564>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    