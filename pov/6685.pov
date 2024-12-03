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
    sphere { m*<-1.0853129750004593,-0.9345151773770646,-0.7658466032280487>, 1 }        
    sphere {  m*<0.3515131705924599,-0.15593665891547195,9.099791140889357>, 1 }
    sphere {  m*<7.706864608592434,-0.2448569349098282,-5.47970214915598>, 1 }
    sphere {  m*<-5.783447998189321,4.811253617628735,-3.1712698527239724>, 1}
    sphere { m*<-2.3473283718189455,-3.5841581945748686,-1.3863689424995>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3515131705924599,-0.15593665891547195,9.099791140889357>, <-1.0853129750004593,-0.9345151773770646,-0.7658466032280487>, 0.5 }
    cylinder { m*<7.706864608592434,-0.2448569349098282,-5.47970214915598>, <-1.0853129750004593,-0.9345151773770646,-0.7658466032280487>, 0.5}
    cylinder { m*<-5.783447998189321,4.811253617628735,-3.1712698527239724>, <-1.0853129750004593,-0.9345151773770646,-0.7658466032280487>, 0.5 }
    cylinder {  m*<-2.3473283718189455,-3.5841581945748686,-1.3863689424995>, <-1.0853129750004593,-0.9345151773770646,-0.7658466032280487>, 0.5}

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
    sphere { m*<-1.0853129750004593,-0.9345151773770646,-0.7658466032280487>, 1 }        
    sphere {  m*<0.3515131705924599,-0.15593665891547195,9.099791140889357>, 1 }
    sphere {  m*<7.706864608592434,-0.2448569349098282,-5.47970214915598>, 1 }
    sphere {  m*<-5.783447998189321,4.811253617628735,-3.1712698527239724>, 1}
    sphere { m*<-2.3473283718189455,-3.5841581945748686,-1.3863689424995>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3515131705924599,-0.15593665891547195,9.099791140889357>, <-1.0853129750004593,-0.9345151773770646,-0.7658466032280487>, 0.5 }
    cylinder { m*<7.706864608592434,-0.2448569349098282,-5.47970214915598>, <-1.0853129750004593,-0.9345151773770646,-0.7658466032280487>, 0.5}
    cylinder { m*<-5.783447998189321,4.811253617628735,-3.1712698527239724>, <-1.0853129750004593,-0.9345151773770646,-0.7658466032280487>, 0.5 }
    cylinder {  m*<-2.3473283718189455,-3.5841581945748686,-1.3863689424995>, <-1.0853129750004593,-0.9345151773770646,-0.7658466032280487>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    