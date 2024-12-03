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
    sphere { m*<-0.6972237625381843,-1.0389574487348197,-0.5724396494351971>, 1 }        
    sphere {  m*<0.7219437316619782,-0.04901853485490193,9.276850447599955>, 1 }
    sphere {  m*<8.089730929984771,-0.334110785647165,-5.293826981473978>, 1 }
    sphere {  m*<-6.806232263704212,6.188970587973489,-3.8030200782923735>, 1}
    sphere { m*<-2.620408926538689,-5.227279218994283,-1.463042754018935>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7219437316619782,-0.04901853485490193,9.276850447599955>, <-0.6972237625381843,-1.0389574487348197,-0.5724396494351971>, 0.5 }
    cylinder { m*<8.089730929984771,-0.334110785647165,-5.293826981473978>, <-0.6972237625381843,-1.0389574487348197,-0.5724396494351971>, 0.5}
    cylinder { m*<-6.806232263704212,6.188970587973489,-3.8030200782923735>, <-0.6972237625381843,-1.0389574487348197,-0.5724396494351971>, 0.5 }
    cylinder {  m*<-2.620408926538689,-5.227279218994283,-1.463042754018935>, <-0.6972237625381843,-1.0389574487348197,-0.5724396494351971>, 0.5}

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
    sphere { m*<-0.6972237625381843,-1.0389574487348197,-0.5724396494351971>, 1 }        
    sphere {  m*<0.7219437316619782,-0.04901853485490193,9.276850447599955>, 1 }
    sphere {  m*<8.089730929984771,-0.334110785647165,-5.293826981473978>, 1 }
    sphere {  m*<-6.806232263704212,6.188970587973489,-3.8030200782923735>, 1}
    sphere { m*<-2.620408926538689,-5.227279218994283,-1.463042754018935>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7219437316619782,-0.04901853485490193,9.276850447599955>, <-0.6972237625381843,-1.0389574487348197,-0.5724396494351971>, 0.5 }
    cylinder { m*<8.089730929984771,-0.334110785647165,-5.293826981473978>, <-0.6972237625381843,-1.0389574487348197,-0.5724396494351971>, 0.5}
    cylinder { m*<-6.806232263704212,6.188970587973489,-3.8030200782923735>, <-0.6972237625381843,-1.0389574487348197,-0.5724396494351971>, 0.5 }
    cylinder {  m*<-2.620408926538689,-5.227279218994283,-1.463042754018935>, <-0.6972237625381843,-1.0389574487348197,-0.5724396494351971>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    